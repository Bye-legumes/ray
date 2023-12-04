from typing import Optional
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.logical.interfaces import LogicalPlan
from ray.data._internal.logical.operators.input_data_operator import InputData
from ray.data._internal.plan import ExecutionPlan
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.shuffle_and_partition import _ShufflePartitionOp
from ray.data._internal.stats import DatasetStats
from ray.data.block import BlockAccessor

def fast_repartition(blocks: BlockList, num_blocks: int, ctx: Optional[TaskContext] = None):
    from ray.data._internal.execution.legacy_compat import _block_list_to_bundles
    from ray.data.dataset import Dataset, Schema

    ref_bundles = _block_list_to_bundles(blocks, blocks._owned_by_consumer)
    logical_plan = LogicalPlan(InputData(ref_bundles))

    wrapped_ds = Dataset(
        ExecutionPlan(
            blocks,
            DatasetStats(stages={}, parent=None),
            run_by_consumer=blocks._owned_by_consumer,
        ),
        logical_plan=logical_plan,
    )

    # Optimized split calculation considering block sizes
    total_size = sum(block.size_bytes for block in blocks)
    target_block_size = total_size / num_blocks
    current_size = 0
    indices = []
    for i, block in enumerate(blocks):
        current_size += block.size_bytes
        if current_size >= target_block_size:
            indices.append(i)
            current_size = 0

    # Lazy evaluation or streaming of splits
    splits = wrapped_ds.lazy_split_at_indices(indices)

    # Coalesce each split into a single block with optimized reduce task
    reduce_task = cached_remote_fn(_ShufflePartitionOp.reduce).options(num_returns=2)

    # Progress bar handling
    should_close_bar = True
    if ctx is not None and ctx.sub_progress_bar_dict is not None:
        bar_name = "Repartition"
        assert bar_name in ctx.sub_progress_bar_dict, ctx.sub_progress_bar_dict
        reduce_bar = ctx.sub_progress_bar_dict[bar_name]
        should_close_bar = False
    else:
        reduce_bar = ProgressBar("Repartition", position=0, total=len(splits))

    reduce_out = [
        reduce_task.remote(False, None, *s.get_internal_block_refs())
        for s in splits
        if s.num_blocks() > 0
    ]

    owned_by_consumer = blocks._owned_by_consumer

    schema = wrapped_ds.schema(fetch_if_missing=True)
    if isinstance(schema, Schema):
        schema = schema.base_schema

    # Early-release memory
    del splits, blocks, wrapped_ds

    new_blocks, new_metadata = zip(*reduce_out)
    new_blocks, new_metadata = list(new_blocks), list(new_metadata)
    new_metadata = reduce_bar.fetch_until_complete(new_metadata)

    if should_close_bar:
        reduce_bar.close()

    # Handle empty blocks
    if len(new_blocks) < num_blocks:
        empty_blocks, empty_metadata = _create_empty_blocks(schema, num_blocks - len(new_blocks))
        new_blocks += empty_blocks
        new_metadata += empty_metadata

    return BlockList(new_blocks, new_metadata, owned_by_consumer=owned_by_consumer), {}

def _create_empty_blocks(schema, num_empties):
    import pyarrow as pa
    from ray.data._internal.arrow_block import ArrowBlockBuilder
    from ray.data._internal.pandas_block import PandasBlockBuilder, PandasBlockSchema

    if schema is None:
        raise ValueError("Dataset is empty or cleared, can't determine the format of the dataset.")
    elif isinstance(schema, pa.Schema):
        builder = ArrowBlockBuilder()
    elif isinstance(schema, PandasBlockSchema):
        builder = PandasBlockBuilder()

    empty_block = builder.build()
    empty_meta = BlockAccessor.for_block(empty_block).get_metadata(input_files=None, exec_stats=None)
    return zip(*[(ray.put(empty_block), empty_meta) for _ in range(num_empties)])
