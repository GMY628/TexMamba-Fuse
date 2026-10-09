                                                                               
                               
                                                                                

#pragma once

#include <cub/config.cuh>

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>
#include <cub/block/block_raking_layout.cuh>
                                               
#include "uninitialized_copy.cuh"
#include "cub_extra.cuh"

   
                                                                                                                    
   
template <
    int         LENGTH,
    typename    T,
    typename    ReductionOp>
__device__ __forceinline__ T ThreadReverseReduce(const T (&input)[LENGTH], ReductionOp reduction_op) {
    static_assert(LENGTH > 0);
    T retval = input[LENGTH - 1];
    #pragma unroll
    for (int i = LENGTH - 2; i >= 0; --i) { retval = reduction_op(retval, input[i]); }
    return retval;
}

   
                                                                                                                                                                  
   
template <
    int         LENGTH,
    typename    T,
    typename    ScanOp>
__device__ __forceinline__ T ThreadReverseScanInclusive(
    const T (&input)[LENGTH],
    T (&output)[LENGTH],
    ScanOp scan_op,
    const T postfix)
{
    T inclusive = postfix;
    #pragma unroll
    for (int i = LENGTH - 1; i >= 0; --i) {
        inclusive = scan_op(inclusive, input[i]);
        output[i] = inclusive;
    }
}

   
                                                                                                                                                                  
   
template <
    int         LENGTH,
    typename    T,
    typename    ScanOp>
__device__ __forceinline__ T ThreadReverseScanExclusive(
    const T (&input)[LENGTH],
    T (&output)[LENGTH],
    ScanOp scan_op,
    const T postfix)
{
                                                
    T exclusive = postfix;
    T inclusive;
    #pragma unroll
    for (int i = LENGTH - 1; i >= 0; --i) {
        inclusive = scan_op(exclusive, input[i]);
        output[i] = exclusive;
        exclusive = inclusive;
    }
    return inclusive;
}


   
                                                                                                                               
  
                                              
   
template <
    typename    T,                                                  
    int         LOGICAL_WARP_THREADS                                           
    >
struct WarpReverseScan {
                                                                           
                                     
                                                                           

                                                                    
    static constexpr bool IS_ARCH_WARP = (LOGICAL_WARP_THREADS == CUB_WARP_THREADS(0));
                                     
    static constexpr int STEPS = cub::Log2<LOGICAL_WARP_THREADS>::VALUE;
    static_assert(LOGICAL_WARP_THREADS == 1 << STEPS);


                                                                           
                    
                                                                           

                                  
    unsigned int lane_id;

                                                     
    unsigned int warp_id;

                                                           
    unsigned int member_mask;

                                                                           
                   
                                                                           

                   
    explicit __device__ __forceinline__
    WarpReverseScan()
        : lane_id((threadIdx.x & 31))
        , warp_id(IS_ARCH_WARP ? 0 : (lane_id / LOGICAL_WARP_THREADS))
                                                                      
        , member_mask(WarpMask<LOGICAL_WARP_THREADS>(warp_id))
    {
        if (!IS_ARCH_WARP) {
            lane_id = lane_id % LOGICAL_WARP_THREADS;
        }
    }


                 
    __device__ __forceinline__ T Broadcast(
        T               input,                                              
        int             src_lane)                                                              
    {
        return cub::ShuffleIndex<LOGICAL_WARP_THREADS>(input, src_lane, member_mask);
    }


                      
    template <typename ScanOpT>
    __device__ __forceinline__ void InclusiveReverseScan(
        T               input,                                                    
        T               &inclusive_output,                                                                         
        ScanOpT         scan_op)                                          
    {
        inclusive_output = input;
        #pragma unroll
        for (int STEP = 0; STEP < STEPS; STEP++) {
            int offset = 1 << STEP;
            T temp = cub::ShuffleDown<LOGICAL_WARP_THREADS>(
                inclusive_output, offset, LOGICAL_WARP_THREADS - 1, member_mask
            );
                                                   
            inclusive_output = static_cast<int>(lane_id) >= LOGICAL_WARP_THREADS - offset
                ? inclusive_output : scan_op(temp, inclusive_output);
        }
    }

                      
                                   
    template <typename ScanOpT>
    __device__ __forceinline__ void ExclusiveReverseScan(
        T              input,                                                    
        T              &exclusive_output,                                                                         
        ScanOpT        scan_op,                                          
        T              &warp_aggregate)                                                            
    {
        T inclusive_output;
        InclusiveReverseScan(input, inclusive_output, scan_op);
        warp_aggregate = cub::ShuffleIndex<LOGICAL_WARP_THREADS>(inclusive_output, 0, member_mask);
                                
        exclusive_output = cub::ShuffleDown<LOGICAL_WARP_THREADS>(
            inclusive_output, 1, LOGICAL_WARP_THREADS - 1, member_mask
        );
    }

       
                                                                                                                                                                                                                                                        
       
    template <typename ScanOpT>
    __device__ __forceinline__ void ReverseScan(
        T               input,                                                    
        T               &inclusive_output,                                                         
        T               &exclusive_output,                                                         
        ScanOpT         scan_op)                                          
    {
        InclusiveReverseScan(input, inclusive_output, scan_op);
                                
        exclusive_output = cub::ShuffleDown<LOGICAL_WARP_THREADS>(
            inclusive_output, 1, LOGICAL_WARP_THREADS - 1, member_mask
        );
    }

};

   
                                                                                                              
   
template <
    typename    T,                                          
    int         BLOCK_DIM_X,                                                                 
    bool        MEMOIZE=false                                                                                                                                         
    >
struct BlockReverseScan {
                                                                           
                          
                                                                           

                 
                                        
    static constexpr int BLOCK_THREADS = BLOCK_DIM_X;

                                                       
    using BlockRakingLayout = cub::BlockRakingLayout<T, BLOCK_THREADS>;
                                                                                                 
    static_assert(BlockRakingLayout::UNGUARDED);

                                
    static constexpr int RAKING_THREADS = BlockRakingLayout::RAKING_THREADS;
                                                                    
    static constexpr int SEGMENT_LENGTH = BlockRakingLayout::SEGMENT_LENGTH;
                                                         
    static constexpr bool WARP_SYNCHRONOUS = (int(BLOCK_THREADS) == int(RAKING_THREADS));

                                     
    using WarpReverseScan = WarpReverseScan<T, RAKING_THREADS>;

                                         
    struct _TempStorage {
        typename BlockRakingLayout::TempStorage raking_grid;                                         
    };


                                                    
    struct TempStorage : cub::Uninitialized<_TempStorage> {};


                                                                           
                        
                                                                           

                    
    _TempStorage    &temp_storage;
    unsigned int    linear_tid;
    T               cached_segment[SEGMENT_LENGTH];


                                                                           
                      
                                                                           

                                                                  
    template <typename ScanOp>
    __device__ __forceinline__ T Upsweep(ScanOp scan_op) {
        T *smem_raking_ptr = BlockRakingLayout::RakingPtr(temp_storage.raking_grid, linear_tid);
                                   
        #pragma unroll
        for (int i = 0; i < SEGMENT_LENGTH; ++i) { cached_segment[i] = smem_raking_ptr[i]; }
        T raking_partial = cached_segment[SEGMENT_LENGTH - 1];
        #pragma unroll
        for (int i = SEGMENT_LENGTH - 2; i >= 0; --i) {
            raking_partial = scan_op(raking_partial, cached_segment[i]);
        }
        return raking_partial;
    }


                                                
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveDownsweep(
        ScanOp          scan_op,
        T               raking_partial)
    {
        T *smem_raking_ptr = BlockRakingLayout::RakingPtr(temp_storage.raking_grid, linear_tid);
                                        
        if (!MEMOIZE) {
            #pragma unroll
            for (int i = 0; i < SEGMENT_LENGTH; ++i) { cached_segment[i] = smem_raking_ptr[i]; }
        }
        ThreadReverseScanExclusive(cached_segment, cached_segment, scan_op, raking_partial);
                                  
        #pragma unroll
        for (int i = 0; i < SEGMENT_LENGTH; ++i) { smem_raking_ptr[i] = cached_segment[i]; }
    }


                                                                           
                   
                                                                           

                   
    __device__ __forceinline__ BlockReverseScan(
        TempStorage &temp_storage)
    :
        temp_storage(temp_storage.Alias()),
        linear_tid(cub::RowMajorTid(BLOCK_DIM_X, 1, 1))
    {}


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    template <
        typename ScanOp,
        typename BlockPostfixCallbackOp>
    __device__ __forceinline__ void ExclusiveReverseScan(
        T                       input,                                                               
        T                       &exclusive_output,                                                                                  
        ScanOp                  scan_op,                                                      
        BlockPostfixCallbackOp  &block_postfix_callback_op)                                                                                                                                                    
    {
        if (WARP_SYNCHRONOUS) {
                                                              
            T block_aggregate;
            WarpReverseScan warp_scan;
            warp_scan.ExclusiveReverseScan(input, exclusive_output, scan_op, block_aggregate);
                                                                               
            T block_postfix = block_postfix_callback_op(block_aggregate);
            block_postfix = warp_scan.Broadcast(block_postfix, 0);
            exclusive_output = linear_tid == BLOCK_THREADS - 1 ? block_postfix : scan_op(block_postfix, exclusive_output);
        } else {
                                                                  
            T *placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid);
            detail::uninitialized_copy(placement_ptr, input);
            __syncthreads();
                                                             
            if (linear_tid < RAKING_THREADS) {
                WarpReverseScan warp_scan;
                                                                  
                T upsweep_partial = Upsweep(scan_op);
                                        
                T exclusive_partial, block_aggregate;
                warp_scan.ExclusiveReverseScan(upsweep_partial, exclusive_partial, scan_op, block_aggregate);
                                                                                    
                T block_postfix = block_postfix_callback_op(block_aggregate);
                block_postfix = warp_scan.Broadcast(block_postfix, 0);
                                                                 
                T downsweep_postfix = linear_tid == RAKING_THREADS - 1
                    ? block_postfix : scan_op(block_postfix, exclusive_partial);
                                                  
                ExclusiveDownsweep(scan_op, downsweep_postfix);
            }
            __syncthreads();
                                                     
            exclusive_output = *placement_ptr;

                                                 
                                                                                  
                                  
                                         
                                                                                         

                                                                                                                                   
                                 
                                                                                               

                                                          
                                  
                                                                              
                                                                         
                

                                                                                                         
                                  
                                                                                
                                      
                                                            
                                                                              
                                                                

                                                                                                              
                    
                

                               

                                                               
                                                            
                                                                                                   
        }
    }


       
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
       
    template <
        int             ITEMS_PER_THREAD,
        typename        ScanOp,
        typename        BlockPostfixCallbackOp>
    __device__ __forceinline__ void InclusiveReverseScan(
        T                       (&input)[ITEMS_PER_THREAD],                                           
        T                       (&output)[ITEMS_PER_THREAD],                                                                         
        ScanOp                  scan_op,                                                     
        BlockPostfixCallbackOp   &block_postfix_callback_op)                                                                                                                                                            
    {
                                                       
        T thread_postfix = ThreadReverseReduce(input, scan_op);
                                      
        ExclusiveReverseScan(thread_postfix, thread_postfix, scan_op, block_postfix_callback_op);
                                                           
        ThreadReverseScanInclusive(input, output, scan_op, thread_postfix);
    }

};