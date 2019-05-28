//=================================================================================================
/*!
//  \file blaze/util/Memory.h
//  \brief Header file for CUDA memory allocation and deallocation functionality
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
//  Copyright (C) 2019 Jules Penuchot - All Rights Reserved
//
//  This file is part of the Blaze library. You can redistribute it and/or modify it under
//  the terms of the New (Revised) BSD License. Redistribution and use in source and binary
//  forms, with or without modification, are permitted provided that the following conditions
//  are met:
//
//  1. Redistributions of source code must retain the above copyright notice, this list of
//     conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice, this list
//     of conditions and the following disclaimer in the documentation and/or other materials
//     provided with the distribution.
//  3. Neither the names of the Blaze development group nor the names of its contributors
//     may be used to endorse or promote products derived from this software without specific
//     prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
//  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
//  SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
//  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//  DAMAGE.
*/
//=================================================================================================

#ifndef _BLAZE_CUDA_UTIL_MEMORY_H_
#define _BLAZE_CUDA_UTIL_MEMORY_H_

#include <cuda_runtime.h>

#include <new>
#include <blaze/util/Assert.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/Exception.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsBuiltin.h>

#include <blaze_cuda/util/CUDAErrorManagement.h>

namespace blaze {

//=================================================================================================
//
//  BACKEND ALLOCATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation for CUDA managed array allocation.
// \ingroup util
//
// \param size The number of bytes to be allocated.
// \return Byte pointer to the first element of the array.
// \exception std::bad_alloc Allocation failed.
//
// This function provides the functionality to allocate CUDA managed memory.
*/
inline byte_t* cuda_managed_allocate_backend( size_t size )
{
   void* raw( nullptr );

   cudaMallocManaged( &raw, size );

   if( cudaGetLastError() != cudaSuccess )
      BLAZE_THROW_BAD_ALLOC;

   return reinterpret_cast<byte_t*>( raw );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation for the deallocation of CUDA memory.
// \ingroup util
//
// \param address The address of the first element of the array to be deallocated.
// \return void
//
// This function deallocates the given memory that was previously allocated via the
// cuda_managed_allocate() function.
*/
inline void cuda_deallocate_backend( const void* address ) noexcept
{
   cudaFree( const_cast<void*>( address ) );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ALLOCATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Allocation for built-in data types.
// \ingroup util
//
// \param size The number of elements of the given type to allocate.
// \return Pointer to the first element of the aligned array.
// \exception std::bad_alloc Allocation failed.
//
// The allocate() function provides the functionality to allocate CUDA managed memory.
*/
template< typename T >
EnableIf_t< IsBuiltin_v<T>, T* > cuda_managed_allocate( size_t size )
{
   return reinterpret_cast<T*>( cuda_managed_allocate_backend( size*sizeof(T) ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Aligned array allocation for user-specific class types.
// \ingroup util
//
// \param size The number of elements of the given type to allocate.
// \return Pointer to the first element of the aligned array.
// \exception std::bad_alloc Allocation failed.
//
// The cuda_managed_allocate() function provides the functionality to allocate CUDA managed memory.
// All elements of the array are guaranteed to be default constructed.
// Note that the cuda_managed_allocate() function provides exception safety similar to the
// new operator: In case any element throws an exception during construction, all elements
// that have already been constructed are destroyed in reverse order and the allocated memory
// is deallocated again.
*/
template< typename T >
DisableIf_t< IsBuiltin_v<T>, T* > cuda_managed_allocate( size_t size )
{
   byte_t* const raw( cuda_managed_allocate_backend( size*sizeof(T) ) );

   *reinterpret_cast<size_t*>( raw ) = size;

   T* const address( reinterpret_cast<T*>( raw ) );
   size_t i( 0UL );

   try {
      for( ; i<size; ++i )
         ::new (address+i) T();
   }
   catch( ... ) {
      while( i != 0UL )
         address[--i].~T();
      cuda_deallocate_backend( raw );
      throw;
   }

   return address;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deallocation of memory for built-in data types.
// \ingroup util
//
// \param address The address of the first element of the array to be deallocated.
// \return void
//
// This function deallocates the given memory that was previously allocated via the
// cuda_managed_deallocate() function.
*/
template< typename T >
EnableIf_t< IsBuiltin_v<T> > cuda_managed_deallocate( T* address ) noexcept
{
   if( address == nullptr )
      return;

   cuda_deallocate_backend( address );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deallocation of memory for user-specific class types.
// \ingroup util
//
// \param address The address of the first element of the array to be deallocated.
// \return void
//
// This function deallocates the given memory that was previously allocated via the
// cuda_managed_deallocate() function.
*/
template< typename T >
DisableIf_t< IsBuiltin_v<T> > cuda_managed_deallocate( T* address )
{
   if( address == nullptr )
      return;

   const byte_t* const raw = reinterpret_cast<byte_t*>( address );

   const size_t size( *reinterpret_cast<const size_t*>( raw ) );
   for( size_t i=0UL; i<size; ++i )
      address[i].~T();

   cudaFree( reinterpret_cast<void*>( address ) );
}
//*************************************************************************************************

} // namespace blaze

#endif
