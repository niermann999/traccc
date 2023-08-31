/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/alpaka/seeding/seeding_algorithm.hpp"
#include "traccc/alpaka/seeding/track_params_estimation.hpp"
#include "traccc/alpaka/utils/definitions.hpp"
#include "traccc/definitions/common.hpp"
#include "traccc/efficiency/nseed_performance_writer.hpp"
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/efficiency/track_filter.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_spacepoints.hpp"
#include "traccc/options/common_options.hpp"
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/seeding_input_options.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// VecMem include(s).
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#endif

#include <vecmem/memory/host_memory_resource.hpp>

// ACTS include(s).
#include <Acts/Definitions/Units.hpp>

// System include(s).
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>

namespace po = boost::program_options;

int seq_run(const traccc::seeding_input_config& /*i_cfg*/,
            const traccc::common_options& common_opts, bool run_cpu) {

    // Read the surface transforms
    auto surface_transforms =
        traccc::io::read_geometry(common_opts.detector_file);

    // Output stats
    uint64_t n_modules = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_seeds = 0;
    uint64_t n_seeds_alpaka = 0;

    // Configs
    traccc::seedfinder_config finder_config;
    traccc::spacepoint_grid_config grid_config(finder_config);
    traccc::seedfilter_config filter_config;

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    vecmem::cuda::copy copy;
    vecmem::cuda::host_memory_resource cuda_host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &cuda_host_mr};
#else
    vecmem::copy copy;
    traccc::memory_resource mr{host_mr, &host_mr};
#endif

    traccc::seeding_algorithm sa(finder_config, grid_config, filter_config,
                                 host_mr);
    traccc::track_params_estimation tp(host_mr);

    // Alpaka Spacepoint Binning
    traccc::alpaka::seeding_algorithm sa_alpaka{finder_config, grid_config,
                                                filter_config, mr, copy};
    traccc::alpaka::track_params_estimation tp_alpaka{mr, copy};

    // performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});

    traccc::nseed_performance_writer nsd_performance_writer(
        "nseed_performance_",
        std::make_unique<traccc::simple_charged_eta_pt_cut>(
            2.7f, 1.f * traccc::unit<traccc::scalar>::GeV),
        std::make_unique<traccc::stepped_percentage>(0.6f));

    if (common_opts.check_performance) {
        nsd_performance_writer.initialize();
    }

    traccc::performance::timing_info elapsedTimes;

    // Loop over events
    for (unsigned int event = common_opts.skip;
         event < common_opts.events + common_opts.skip; ++event) {

        traccc::io::spacepoint_reader_output reader_output(mr.host);
        traccc::seeding_algorithm::output_type seeds;
        traccc::track_params_estimation::output_type params;

        // Instantiate alpaka containers/collections
        traccc::seed_collection_types::buffer seeds_alpaka_buffer(0,
                                                                  *(mr.host));
        traccc::bound_track_parameters_collection_types::buffer
            params_alpaka_buffer(0, *mr.host);

        {  // Start measuring wall time
            traccc::performance::timer wall_t("Wall time", elapsedTimes);

            /*-----------------
            hit file reading
            -----------------*/
            {
                traccc::performance::timer t("Hit reading  (cpu)",
                                             elapsedTimes);
                // Read the hits from the relevant event file
                traccc::io::read_spacepoints(
                    reader_output, event, common_opts.input_directory,
                    surface_transforms, common_opts.input_data_format);
            }  // stop measuring hit reading timer

            traccc::spacepoint_collection_types::host& spacepoints_per_event =
                reader_output.spacepoints;
            auto& modules_per_event = reader_output.modules;

            /*----------------------------
                Seeding algorithm
            ----------------------------*/

            // Alpaka

            // TODO: Check this (and all other copies) are intelligent.
            // Copy the spacepoint data to the device.
            traccc::spacepoint_collection_types::buffer
                spacepoints_alpaka_buffer(spacepoints_per_event.size(),
                                          mr.main);
            copy(vecmem::get_data(spacepoints_per_event),
                 spacepoints_alpaka_buffer);
            traccc::cell_module_collection_types::buffer modules_buffer(
                modules_per_event.size(), mr.main);
            copy(vecmem::get_data(modules_per_event), modules_buffer);

            {
                traccc::performance::timer t("Seeding (alpaka)", elapsedTimes);
                // Reconstruct the spacepoints into seeds.
                seeds_alpaka_buffer =
                    sa_alpaka(vecmem::get_data(spacepoints_alpaka_buffer));
            }

            // CPU

            if (run_cpu) {
                traccc::performance::timer t("Seeding  (cpu)", elapsedTimes);
                seeds = sa(spacepoints_per_event);
            }  // stop measuring seeding cpu timer

            /*----------------------------
            Track params estimation
            ----------------------------*/

            // Alpaka

            {
                traccc::performance::timer t("Track params (alpaka)",
                                             elapsedTimes);
                params_alpaka_buffer =
                    tp_alpaka(spacepoints_alpaka_buffer, seeds_alpaka_buffer,
                              modules_buffer, {0.f, 0.f, finder_config.bFieldInZ});
            }  // stop measuring track params alpaka timer
            // CPU

            if (run_cpu) {
                traccc::performance::timer t("Track params  (cpu)",
                                             elapsedTimes);
                params =
                    tp(std::move(spacepoints_per_event), seeds,
                       modules_per_event, {0.f, 0.f, finder_config.bFieldInZ});
            }  // stop measuring track params cpu timer

        }  // Stop measuring wall time

        /*----------------------------------
          compare seeds from cpu and alpaka
          ----------------------------------*/

        // Copy the seeds to the host for comparisons
        traccc::seed_collection_types::host seeds_alpaka;
        traccc::bound_track_parameters_collection_types::host params_alpaka;
        copy(seeds_alpaka_buffer, seeds_alpaka);
        copy(params_alpaka_buffer, params_alpaka);

        if (run_cpu) {
            // Show which event we are currently presenting the results for.
            std::cout << "===>>> Event " << event << " <<<===" << std::endl;

            // Compare the seeds made on the host and on the device
            traccc::collection_comparator<traccc::seed> compare_seeds{
                "seeds", traccc::details::comparator_factory<traccc::seed>{
                             vecmem::get_data(reader_output.spacepoints),
                             vecmem::get_data(reader_output.spacepoints)}};
            compare_seeds(vecmem::get_data(seeds),
                          vecmem::get_data(seeds_alpaka));

            // Compare the track parameters made on the host and on the device.
            traccc::collection_comparator<traccc::bound_track_parameters>
                compare_track_parameters{"track parameters"};
            compare_track_parameters(vecmem::get_data(params),
                                     vecmem::get_data(params_alpaka));
        }

        /*----------------
             Statistics
          ---------------*/

        n_spacepoints += reader_output.spacepoints.size();
        n_modules += reader_output.modules.size();
        n_seeds_alpaka += seeds_alpaka.size();
        n_seeds += seeds.size();

        /*------------
          Writer
          ------------*/

        if (common_opts.check_performance) {
            traccc::event_map evt_map(event, common_opts.detector_file,
                                      common_opts.input_directory,
                                      common_opts.input_directory, host_mr);

            std::vector<traccc::nseed<3>> nseeds;

            std::transform(
                seeds_alpaka.cbegin(), seeds_alpaka.cend(),
                std::back_inserter(nseeds),
                [](const traccc::seed& s) { return traccc::nseed<3>(s); });

            nsd_performance_writer.register_event(
                event, nseeds.begin(), nseeds.end(),
                reader_output.spacepoints.begin(), evt_map);

            sd_performance_writer.write(
                vecmem::get_data(seeds_alpaka),
                vecmem::get_data(reader_output.spacepoints), evt_map);
        }
    }

    if (common_opts.check_performance) {
        sd_performance_writer.finalize();
        nsd_performance_writer.finalize();

        std::cout << nsd_performance_writer.generate_report_str();
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_spacepoints << " spacepoints from "
              << n_modules << " modules" << std::endl;
    std::cout << "- created  (cpu)  " << n_seeds << " seeds" << std::endl;
    std::cout << "- created (alpaka)  " << n_seeds_alpaka << " seeds"
              << std::endl;
    std::cout << "==>Elapsed times...\n" << elapsedTimes << std::endl;

    return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {
    // Set up the program options
    po::options_description desc("Allowed options");

    // Add options
    desc.add_options()("help,h", "Give some help with the program's options");
    traccc::common_options common_opts(desc);
    traccc::seeding_input_config seeding_input_cfg(desc);
    desc.add_options()("run_cpu", po::value<bool>()->default_value(false),
                       "run cpu tracking as well");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    common_opts.read(vm);
    seeding_input_cfg.read(vm);
    auto run_cpu = vm["run_cpu"].as<bool>();

    std::cout << "Running " << argv[0] << " " << common_opts.detector_file
              << " " << common_opts.input_directory << " " << common_opts.events
              << std::endl;

    return seq_run(seeding_input_cfg, common_opts, run_cpu);
}
