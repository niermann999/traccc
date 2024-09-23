/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/finding/ckf_aborter.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/finding/interaction_register.hpp"
#include "traccc/fitting/kalman_filter/gain_matrix_updater.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// detray include(s).
#include "detray/propagator/actor_chain.hpp"
#include "detray/propagator/actors/aborters.hpp"
#include "detray/propagator/actors/parameter_resetter.hpp"
#include "detray/propagator/actors/parameter_transporter.hpp"
#include "detray/propagator/actors/pointwise_material_interactor.hpp"
#include "detray/propagator/propagator.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// Thrust Library
#include <thrust/pair.h>

namespace traccc {

/// Track Finding algorithm for a set of tracks
template <typename stepper_t, typename navigator_t>
class finding_algorithm
    : public algorithm<track_candidate_container_types::host(
          const typename navigator_t::detector_type&,
          const typename stepper_t::magnetic_field_type&,
          const measurement_collection_types::host&,
          const bound_track_parameters_collection_types::host&)> {

    /// Detector type
    using detector_type = typename navigator_t::detector_type;
    using cxt_t = typename detector_type::geometry_context;

    /// Algebra types
    using algebra_type = typename detector_type::algebra_type;
    /// scalar type
    using scalar_type = detray::dscalar<algebra_type>;

    /// Actor types
    using aborter = detray::pathlimit_aborter;
    using transporter = detray::parameter_transporter<algebra_type>;
    using interactor = detray::pointwise_material_interactor<algebra_type>;
    using resetter = detray::parameter_resetter<algebra_type>;

    using actor_type = detray::actor_chain<detray::tuple, aborter, transporter,
                                           interactor, resetter>;

    using propagator_type =
        detray::propagator<stepper_t, navigator_t, actor_type>;

    using interactor_type = detray::pointwise_material_interactor<algebra_type>;

    using bfield_type = typename stepper_t::magnetic_field_type;

    // Trace the measurements and number of holes per track
    struct trace_state {
        unsigned int n_skipped{0u};
        vecmem::vector<track_candidate> meas_trace{};
    };

    /// Package measurements with the corresponding filtered track parameters
    struct candidate {
        bound_track_parameters filtered_params;
        unsigned int meas_id;
        float chi2;

        /// @param rhs is the right hand side candidate for comparison
        constexpr bool operator<(const candidate& rhs) const {
            return (chi2 < rhs.chi2);
        }

        /// @param rhs is the left hand side candidate for comparison
        constexpr bool operator>(const candidate& rhs) const {
            return (chi2 > rhs.chi2);
        }
    };

    public:
    /// Configuration type
    using config_type = finding_config;

    /// Constructor for the finding algorithm
    ///
    /// @param cfg  Configuration object
    /// @param mr   The memory resource to use
    finding_algorithm(const config_type& cfg) : m_cfg(cfg) {}

    /// Get config object (const access)
    const finding_config& get_config() const { return m_cfg; }

    /// Run the algorithm
    ///
    /// @param det    Detector
    /// @param measurements  Input measurements
    /// @param seeds  Input seeds
    track_candidate_container_types::host operator()(
        const detector_type& det, const bfield_type& field,
        const measurement_collection_types::host& measurements,
        const bound_track_parameters_collection_types::host& seeds) const;

    private:
    /// Config object
    config_type m_cfg;
};

}  // namespace traccc

//#include "traccc/finding/finding_algorithm.ipp"
#include "traccc/finding/finding_algorithm_alt.ipp"
