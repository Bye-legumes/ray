from typing import Any, Dict, List, Optional, Tuple
import ray
from ray.data._internal.execution.interfaces import RefBundle, TaskContext
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskScheduler
from ray.data._internal.planner.exchange.shuffle_task_spec import ShuffleTaskSpec
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.split import _split_at_indices
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.types import ObjectRef

class SplitRepartitionTaskScheduler(ExchangeTaskScheduler):
    def execute(
        self,
        refs: List[RefBundle],
        output_num_blocks: int,
        ctx: TaskContext,
        map_ray_remote_args: Optional[Dict[str, Any]] = None,
        reduce_ray_remote_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[RefBundle], StatsDict]:
        input_num_rows = sum(ref_bundle.num_rows() for ref_bundle in refs)
        input_owned_by_consumer = all(ref_bundle.owns_blocks for ref_bundle in refs)

        indices = self.optimized_split_indices(refs, output_num_blocks)

        blocks_with_metadata = [block for ref_bundle in refs for block in ref_bundle.blocks]
        split_return = self.streamlined_split_at_indices(blocks_with_metadata, indices, input_owned_by_consumer)
        del blocks_with_metadata

        split_block_refs, split_metadata = zip(*split_return)
        del split_return

        reduce_task = cached_remote_fn(self.optimized_reduce)
        reduce_return = [
            reduce_task.options(**reduce_ray_remote_args, num_returns=2).remote(
                *split_block_refs[j],
            )
            for j in range(output_num_blocks)
            if split_block_refs[j]
        ]
        del split_block_refs

        reduce_block_refs, reduce_metadata = zip(*reduce_return)
        del reduce_return

        output = [RefBundle([(block, meta)], owns_blocks=input_owned_by_consumer) for block, meta in zip(reduce_block_refs, reduce_metadata)]
        del reduce_block_refs, reduce_metadata

        stats = {"split": split_metadata, "reduce": reduce_metadata}
        return output, stats

    def optimized_split_indices(self, refs: List[RefBundle], output_num_blocks: int) -> List[int]:
        total_rows = sum(ref_bundle.num_rows() for ref_bundle in refs)
        rows_per_block = total_rows // output_num_blocks
        indices = []
        current_row_count = 0

        for ref_bundle in refs:
            current_row_count += ref_bundle.num_rows()
            if current_row_count >= rows_per_block:
                indices.append(current_row_count)
                current_row_count = 0

        return indices

    def streamlined_split_at_indices(self, blocks_with_metadata, indices, input_owned_by_consumer):
        current_index = 0
        for block, metadata in blocks_with_metadata:
            block_accessor = BlockAccessor.for_block(block)
            while current_index < len(indices) and block_accessor.num_rows() > 0:
                split_index = indices[current_index] - block_accessor.start_row()
                if split_index >= block_accessor.num_rows():
                    yield block, metadata
                    current_index += 1
                else:
                    split_block, remaining_block = block_accessor.split(split_index)
                    yield split_block, metadata
                    block_accessor = BlockAccessor.for_block(remaining_block)

    def optimized_reduce(self, *mapper_outputs: List[Block], partial_reduce: bool = False) -> (Block, BlockMetadata):
        from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
        from ray.data.block import BlockAccessor, BlockMetadata

        builder = DelegatingBlockBuilder()
        for block in mapper_outputs:
            builder.add_block(block)

        combined_block = builder.build()
        accessor = BlockAccessor.for_block(combined_block)

        new_metadata = BlockMetadata(
            num_rows=accessor.num_rows(),
            size_bytes=accessor.size_bytes(),
            schema=accessor.schema(),
            input_files=None,
            exec_stats=None,
        )

        return combined_block, new_metadata
