/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/containers/vector.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/data/jagged_vector_buffer.hpp>
#include <vecmem/containers/data/vector_buffer.hpp>

namespace traccc {

    /// Container describing objects in a given event
    ///
    /// This is the generic container of the code, holding all relevant
    /// information about objcts in a given event.
    ///
    /// It can be instantiated with different vector types, to be able to use
    /// the same container type in both host and device code.
    ///
    /// It also can be instantiated with different edm types represented by 
    /// header and item type.
    template< typename header_t,
	      typename item_t,
	      template< typename > class vector_t,
              template< typename > class jagged_vector_t >
    class container{
    public:

        /// @name Type definitions
        /// @{

        /// Vector type used by the container	
        template< typename T >
        using vector_type = vector_t< T >;
	
	/// Jagged vector type used by the container
        template< typename T >
        using jagged_vector_type = jagged_vector_t< T >;

	/// The header vector type
        using header_vector = vector_type< header_t >;

	/// The item vector type
        using item_vector = jagged_vector_type< item_t >;
	
        /// @}

        /// Headers information related to the objects in the event
	header_vector headers;

	/// All objects in the event
	item_vector items;
    };

    /// Convenience declaration for the container type to use in host code    
    template< typename header_t, typename item_t >
    using host_container = container< header_t, item_t,
				      vecmem::vector, vecmem::jagged_vector >;

    /// Convenience declaration for the container type to use in device code     
    template< typename header_t, typename item_t >
    using device_container = container< header_t, item_t,
					vecmem::device_vector, vecmem::jagged_device_vector >;

    /// @name Types used to send data back and forth between host and device code
    /// @{

    /// Structure holding (some of the) data about the container in host code
    template< typename header_t, typename item_t >
    struct container_data{
	vecmem::data::vector_view< header_t > headers;
	vecmem::data::jagged_vector_data< item_t > items;
    };

    /// Structure holding (all of the) data about the container in host code    
    template< typename header_t, typename item_t >
    struct container_buffer{
	vecmem::data::vector_buffer< header_t > headers;
	vecmem::data::jagged_vector_buffer< item_t > items;
    };

    /// Structure used to send the data about the container to device code
    ///
    /// This is the type that can be passed to device code as-is. But since in
    /// host code one needs to manage the data describing a
    /// @c traccc::container either using @c traccc::container_data or
    /// @c traccc::container_buffer, it needs to have constructors from
    /// both of those types.
    ///
    /// In fact it needs to be created from one of those types, as such an
    /// object can only function if an instance of one of those types exists
    /// alongside it as well.
    ///    
    template< typename header_t, typename item_t >
    struct container_view{

	/// Constructor from a @c container_data object	
        container_view( const container_data< header_t, item_t >& data )
	    : headers( data.headers ), items( data.items ) {}
	
        /// Constructor from a @c container_buffer object
        container_view( const container_buffer< header_t, item_t >& buffer )
        : headers( buffer.headers ), items( buffer.items ) {}

	/// View of the data describing the headers
        vecmem::data::vector_view< header_t > headers;

	/// View of the data describing the items	
        vecmem::data::jagged_vector_view< item_t > items;

    };

    /// Helper function for making a "simple" object out of the container
    template< typename header_t, typename item_t >
    __inline__
    container_data<header_t, item_t> get_data( host_container< header_t, item_t >& cc, vecmem::memory_resource* resource = nullptr ) {
        return { { vecmem::get_data( cc.headers ) },
                 { vecmem::get_data( cc.items, resource ) } };
    }
    
}
