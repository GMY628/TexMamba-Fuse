                                                                               
                                                                     
  
                                                                     
                                                                              
                                                                       
                                                                      
                                                                          
                                                                            
                                                                             
                                                           
                                                                             
                                                                              
  
                                                                              
                                                                            
                                                                             
                                                                         
                                                                             
                                                                               
                                                                              
                                                                             
                                                                                
                                                               
  
                                                                                

#pragma once

#include <cub/config.cuh>

#include <cuda/std/type_traits>


namespace detail
{

#if defined(_NVHPC_CUDA)
template <typename T, typename U>
__host__ __device__ void uninitialized_copy(T *ptr, U &&val)
{
                  
  new (ptr) T(::cuda::std::forward<U>(val));
}
#else
template <typename T,
          typename U,
          typename ::cuda::std::enable_if<
            ::cuda::std::is_trivially_copyable<T>::value,
            int
          >::type = 0>
__host__ __device__ void uninitialized_copy(T *ptr, U &&val)
{
  *ptr = ::cuda::std::forward<U>(val);
}

template <typename T,
         typename U,
         typename ::cuda::std::enable_if<
           !::cuda::std::is_trivially_copyable<T>::value,
           int
         >::type = 0>
__host__ __device__ void uninitialized_copy(T *ptr, U &&val)
{
  new (ptr) T(::cuda::std::forward<U>(val));
}
#endif

}                    
