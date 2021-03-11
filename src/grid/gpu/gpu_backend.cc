/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2021 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

#ifdef __GRID_CUDA

#include "../common/grid_context.hpp"
#include "headers/gpu_backend.hpp"

void gpu_context::collocate_one_grid_level(const int level)
{
    if (ctx_.tasks_per_level(level) == 0)
        return;

    prepare_ldiffs ldiffs = prepare_get_ldiffs(ctx_.func());
    int lmax[2] = {ctx_.lmax_ + ldiffs.la_max_diff, ctx_.lmax_ + ldiffs.lb_max_diff};

    /* all devices have their own work space so parallelization is trivial here */
    for (auto &dev : dev_) {
        acc::SetDevice(dev.device_id_);
        dev.collocate_one_grid_level(*this, level);
    }
    // need to implement the reduction over multiple devices
}

void gpu_context::integrate_one_grid_level(const int level)
{
    if (ctx_.tasks_per_level(level) == 0)
        return;

    /* all devices have their own work space so parallelization is trivial here */
    for (auto &dev : dev_) {
        acc::SetDevice(dev.device_id_);
        dev.integrate_one_grid_level(*this, level);
        //dev.synchronize();
    }

    // need to implement the reduction over multiple devices
}


void gpu_context::collocate() {
  // Upload blocks buffer using the main stream
  for (auto &dev : dev_) {
      acc::SetDevice(dev.device_id_);
      dev.copy_pab_blocks(ctx_.pab_blocks());
  }

  for (auto &dev : dev_) {
      for (int level = 0; level < ctx_.grid().size(); level += dev_.size()) {
          dev.set_grid_parameters(ctx_.grid(level));
          this->collocate_one_grid_level(level);
      }
  }

  for (auto &dev : dev_) {
      dev.clear();
  }
}

void gpu_context::integrate() {
    // Upload blocks buffer using the main stream
    for (auto &dev : dev_) {
        acc::SetDevice(dev.device_id_);
        dev.copy_pab_blocks(ctx_.pab_blocks());
        dev.allocate_hab_blocks(ctx_.hab_blocks());

        if (ctx_.calculate_forces()) {
            dev.forces_dev_ = static_cast<double*>(acc::DeviceMalloc(ctx_.forces().size() * sizeof(double)));
            acc::device_memset_async(dev.forces_dev_, 0, sizeof(double) * ctx_.forces().size(), dev.stream_);
        }

        if (ctx_.calculate_virial()) {
            dev.virial_dev_ = static_cast<double*>(acc::DeviceMalloc(ctx_.virial().size()));
            acc::device_memset_async(dev.virial_dev_, 0, sizeof(double) * ctx_.virial().size(), dev.stream_);
        }
    }

    for (auto &dev : dev_) {
        for (int level = 0; level < ctx_.grid().size(); level += dev_.size()) {
            dev.set_grid_parameters(ctx_.grid(level));
            this->integrate_one_grid_level(level);
        }
    }

    for (auto &dev : dev_) {
        dev.copy_hab_blocks(ctx_.hab_blocks());
    }

    if (ctx_.calculate_forces()) {
        for (auto &dev : dev_)
            acc::CopyFromDeviceAsync(ctx_.forces().at(), dev.forces_dev_, sizeof(double) * ctx_.forces().size(), dev.stream_);
    }

    if (ctx_.calculate_virial()) {
        for (auto &dev : dev_)
            acc::CopyFromDeviceAsync(ctx_.virial().at(), dev.virial_dev_, sizeof(double) * ctx_.virial().size(), dev.stream_);
    }

    for (auto &dev : dev_) {
        cudaStreamSynchronize(dev.stream_);
        dev.clear();
    }
}
#endif
