#ifndef GPU_CONTEXT_HPP
#define GPU_CONTEXT_HPP
#include <vector>

extern "C" {
#include "../../common/grid_basis_set.h"
#include "../../common/grid_prepare_pab.h"
};

#include "../../common/task.hpp"
#include "../../common/grid_context.hpp"
#include "../cuda/acc.hpp"
class gpu_context;
class device {
public:
    task_info *task_dev_{nullptr};
    std::vector<grid_basis_set> basis_sets_host_;
    grid_basis_set *basis_sets_dev_{nullptr};
    // double *coef_dev_{nullptr};
    // double *pab_dev_{nullptr};
    // double *pab_prep_dev_{nullptr};
    double *pab_blocks_dev_{nullptr};
    double *hab_blocks_dev_{nullptr};
    int *block_offsets_dev_{nullptr};
    size_t pab_block_size_{0};
    size_t hab_block_size_{0};
    double *grid_dev_{nullptr};
    double *forces_dev_{nullptr};
    double *virial_dev_{nullptr};
    int device_id_{-1};
    int lmin_diff[2] = {0, 0};
    int lmax_diff[2] = {0, 0};
    int lb_max_diff_{0};
    int smem_cab_offset_{0};
    int smem_alpha_offset_{0};
    int smem_cxyz_offset_{0};
    enum grid_func func_;
    bool orthorhombic_{false};
    int first_task_{0};
    int grid_lower_corner_[3] = {0,0,0};
    int grid_local_size_[3] = {0,0,0};
    int grid_full_size_[3] = {0,0,0};
    int grid_border_width_[3] = {0, 0, 0};
    int lmax_{0};
    double dh_[9];
    double dh_inv_[9];
    cudaStream_t stream_;
    int blocks_offsets_size_ = {0};
private:
    int grid_size_{0};
    int grid_alloc_size_{0};
    int task_list_length_{0};
public:
    device() {};

    void initialize_device(int device_id__)
        {
            device_id_ = device_id__;
            acc::SetDevice(device_id_);
            acc::create_stream(&stream_);
        }

    ~device() {};

    void release_gpu_resources() {
        acc::SetDevice(device_id_);
        acc::destroy_stream(stream_);

        if (task_dev_) {
            acc::DeviceFree(task_dev_);
        }

        // if (coef_dev_) {
        //     acc::DeviceFree(coef_dev_);
        // }

        if (pab_blocks_dev_) {
            acc::DeviceFree(pab_blocks_dev_);
        }

        if (hab_blocks_dev_) {
            acc::DeviceFree(hab_blocks_dev_);
        }

        // if (work_dev_) {
        //     acc::DeviceFree(work_dev_);
        // }

        // if (pab_dev) {
        //     acc::DeviceFree(pab_dev_);
        //     }

        // if (pab_prep_dev) {
        //     acc::DeviceFree(pab_prep_dev_);
        // }

        if (forces_dev_) {
            acc::DeviceFree(forces_dev_);
            forces_dev_ = nullptr;
        }

        if (virial_dev_) {
            acc::DeviceFree(virial_dev_);
            virial_dev_ = nullptr;
        }

        if (grid_dev_) {
            acc::DeviceFree(grid_dev_);
            grid_dev_ = nullptr;
        }

        for (auto &basis_set : basis_sets_host_) {
            acc::DeviceFree(basis_set.lmin);
            acc::DeviceFree(basis_set.lmax);
            acc::DeviceFree(basis_set.npgf);
            acc::DeviceFree(basis_set.nsgf_set);
            acc::DeviceFree(basis_set.first_sgf);
            acc::DeviceFree(basis_set.sphi);
            acc::DeviceFree(basis_set.zet);
        }

        acc::DeviceFree(basis_sets_dev_);
    }

    void init_basis_sets(std::vector<grid_basis_set*> &set_) {
        for (auto &basis_set : basis_sets_host_) {
            acc::DeviceFree(basis_set.lmin);
            acc::DeviceFree(basis_set.lmax);
            acc::DeviceFree(basis_set.npgf);
            acc::DeviceFree(basis_set.nsgf_set);
            acc::DeviceFree(basis_set.first_sgf);
            acc::DeviceFree(basis_set.sphi);
            acc::DeviceFree(basis_set.zet);
        }

        if (basis_sets_dev_)
            acc::DeviceFree(basis_sets_dev_);

        basis_sets_host_.resize(set_.size());
        for (int kind = 0; kind < set_.size(); kind++) {
            const auto &set = *set_[kind];
            auto &set_host_ = basis_sets_host_[kind];

            set_host_.nset = set.nset;
            set_host_.nsgf = set.nsgf;
            set_host_.maxco = set.maxco;
            set_host_.maxpgf = set.maxpgf;

            /* allocate ressources */
            set_host_.lmin = static_cast<int*>(acc::DeviceMalloc(set.nset * sizeof(int)));
            set_host_.lmax = static_cast<int*>(acc::DeviceMalloc(set.nset * sizeof(int)));
            set_host_.npgf = static_cast<int*>(acc::DeviceMalloc(set.nset * sizeof(int)));
            set_host_.nsgf_set = static_cast<int*>(acc::DeviceMalloc(set.nset * sizeof(int)));
            set_host_.first_sgf = static_cast<int*>(acc::DeviceMalloc(set.nset * sizeof(int)));
            set_host_.sphi = static_cast<double*>(acc::DeviceMalloc(set.nsgf * set.maxco * sizeof(double)));
            set_host_.zet = static_cast<double*>(acc::DeviceMalloc(set.nset * set.maxco * sizeof(double)));

            /* copy data on the device */
            acc::CopyToDevice(set_host_.lmin, set.lmin, set.nset * sizeof(int));
            acc::CopyToDevice(set_host_.lmax, set.lmax, set.nset * sizeof(int));
            acc::CopyToDevice(set_host_.npgf, set.npgf, set.nset * sizeof(int));
            acc::CopyToDevice(set_host_.nsgf_set, set.nsgf_set, set.nset * sizeof(int));
            acc::CopyToDevice(set_host_.first_sgf, set.first_sgf, set.nset * sizeof(int));
            acc::CopyToDevice(set_host_.sphi, set.sphi, set.nsgf * set.maxco * sizeof(double));
            acc::CopyToDevice(set_host_.zet, set.zet, set.nset * set.maxco * sizeof(double));
        }

        basis_sets_dev_ = static_cast<grid_basis_set*>(acc::DeviceMalloc(set_.size() * sizeof(grid_basis_set)));
        acc::CopyToDevice(basis_sets_dev_, &basis_sets_host_[0], set_.size() * sizeof(grid_basis_set));
    }

    void init_task_list(grid_context &ctx_) {
        if (ctx_.task_list().size() > this->task_list_length_) {
            acc::DeviceFree(this->task_dev_);
            this->task_list_length_ = ctx_.task_list().size();
            this->task_dev_ = nullptr;
        }

        this->task_dev_ = static_cast<task_info*>(acc::DeviceMalloc(ctx_.task_list().size() * sizeof(task_info)));
        acc::CopyToDevice(this->task_dev_, ctx_.queues(0, 0), ctx_.task_list().size() * sizeof(task_info));
    }

    void copy_pab_blocks(grid_buffer &pab_blocks) {
        acc::SetDevice(device_id_);

        if (pab_block_size_ < pab_blocks.size) {
            if (pab_blocks_dev_)
                acc::DeviceFree(pab_blocks_dev_);
            pab_blocks_dev_ = static_cast<double*>(acc::DeviceMalloc(pab_blocks.size));
            pab_block_size_ = pab_blocks.size;
        }

        acc::CopyToDeviceAsync(pab_blocks_dev_, pab_blocks.host_buffer, pab_blocks.size, stream_);
    }

    void free_pab_blocks() {
        if (pab_blocks_dev_)
            acc::DeviceFree(pab_blocks_dev_);
        pab_blocks_dev_ = nullptr;
        pab_block_size_ = 0;
    }

    void allocate_hab_blocks(grid_buffer &pab_blocks) {
        if (hab_block_size_ < pab_blocks.size) {
            if (hab_blocks_dev_)
                acc::DeviceFree(hab_blocks_dev_);
            hab_blocks_dev_ = static_cast<double*>(acc::DeviceMalloc(pab_blocks.size));
            hab_block_size_ = pab_blocks.size;
        }
    }
    void copy_hab_blocks(grid_buffer &pab_blocks) {
        acc::SetDevice(device_id_);
        acc::CopyFromDeviceAsync(pab_blocks.host_buffer, hab_blocks_dev_, pab_blocks.size, stream_);
    }

    void free_hab_blocks() {
        if (hab_blocks_dev_)
            acc::DeviceFree(hab_blocks_dev_);
        hab_blocks_dev_ = nullptr;
        hab_block_size_ = 0;
    }

    void allocate_grid(const int size) {
        grid_dev_ = static_cast<double*>(acc::DeviceMalloc(sizeof(double) * size));
    }

    void clear() {
        if (hab_blocks_dev_)
            acc::DeviceFree(hab_blocks_dev_);
        if (pab_blocks_dev_)
            acc::DeviceFree(pab_blocks_dev_);
        if (forces_dev_)
            acc::DeviceFree(forces_dev_);
        if (forces_dev_)
            acc::DeviceFree(virial_dev_);
        if (grid_dev_)
            acc::DeviceFree(grid_dev_);

        hab_blocks_dev_ = nullptr;
        pab_blocks_dev_ = nullptr;
        hab_block_size_ = 0;
        pab_block_size_ = 0;
        grid_dev_ = nullptr;
        grid_alloc_size_ = 0;
        forces_dev_ = nullptr;
        virial_dev_ = nullptr;
        pab_block_size_ = 0;
        hab_block_size_ = 0;
    }

    void set_grid_parameters(const grid_info &grid) {
        const auto &local_size_ = grid.size();
        grid_local_size_[0] = grid.size(0);
        grid_local_size_[1] = grid.size(1);
        grid_local_size_[2] = grid.size(2);

        grid_full_size_[0] = grid.full_size(0);
        grid_full_size_[1] = grid.full_size(1);
        grid_full_size_[2] = grid.full_size(2);

        grid_lower_corner_[0] = grid.lower_corner(0);
        grid_lower_corner_[1] = grid.lower_corner(1);
        grid_lower_corner_[2] = grid.lower_corner(2);

        grid_border_width_[0] = grid.border_width(0);
        grid_border_width_[1] = grid.border_width(1);
        grid_border_width_[2] = grid.border_width(2);

        memcpy(dh_, grid.dh[0], sizeof(double) * 9);
        memcpy(dh_inv_, grid.dh_inv[0], sizeof(double) * 9);

        orthorhombic_ = grid.is_orthorhombic();

        if ((grid_alloc_size_ < grid.size()) || grid_dev_ == nullptr) {
            if (grid_dev_ != nullptr)
                acc::DeviceFree(grid_dev_);
            grid_size_ = grid.size();
            grid_alloc_size_ = grid_size_;
            grid_dev_ = static_cast<double*>(acc::DeviceMalloc(sizeof(double) * grid.size()));
        }
    }

    void set_func(const grid_func func__, const int lmax) {
        func_ = func__;
        const prepare_ldiffs ldiffs = prepare_get_ldiffs(func_);
        lmax_ = lmax;
        lmin_diff[0] = ldiffs.la_min_diff;
        lmin_diff[1] = ldiffs.lb_min_diff;
        lmax_diff[0] = ldiffs.la_max_diff;
        lmax_diff[1] = ldiffs.lb_max_diff;
    }

    void collocate_one_grid_level(gpu_context &ctx_, const int level);
    void integrate_one_grid_level(gpu_context &ctx_, const int level);

    void synchronize() {
        cudaStreamSynchronize(stream_);
    }
};

class gpu_context {
    grid_context &ctx_;
    std::vector<device> dev_;
    bool apply_cutoff_{false};
    std::vector<grid_info> grid_;
    bool orthorhombic_{false};
    enum grid_func func;
public:
    gpu_context (grid_context &ctx__, std::vector<int> &device_id_)
        :ctx_(ctx__) {
        dev_.clear();
        grid_.clear();
        set_number_of_devices(device_id_);
    }

    ~gpu_context() {
        for (auto &dev : dev_) {
            dev.release_gpu_resources();
        }
        dev_.clear();
        grid_.clear();
    }

    grid_context &ctx() {
        return ctx_;
    }

    const grid_context &ctx() const {
        return ctx_;
    }

    inline std::vector<grid_info> &grid() {
        return ctx_.grid();
    }

    inline grid_info &grid(const int level) {
        return ctx_.grid(level);
    }


    void set_number_of_devices(std::vector<int> &device_id_)
        {
            dev_.resize(device_id_.size());

            for (auto i = 0u; i < dev_.size(); i++)
                dev_[i].initialize_device(device_id_[i]);
        }

    void set_grid() {
        grid_ = ctx_.grid();

        int max_size = 0;
        for (auto &grid : grid_) {
            max_size = std::max(grid.size(), max_size);
        }

        for (auto &dev : dev_) {
            dev.allocate_grid(max_size);
        }
    }

    void clear() {
        for (auto &dev : dev_) {
            dev.clear();
        }
    }

    inline void update_block_offsets() {
        for (auto &dev: dev_) {
            if (dev.blocks_offsets_size_ < ctx_.block_offsets_.size()) {
                acc::SetDevice(dev.device_id_);
                if (dev.block_offsets_dev_)
                    acc::DeviceFree(dev.block_offsets_dev_);
                dev.block_offsets_dev_ = nullptr;
                dev.blocks_offsets_size_ = ctx_.block_offsets_.size();
                dev.block_offsets_dev_ = static_cast<int*>(acc::DeviceMalloc(sizeof(int) * ctx_.block_offsets_.size()));
            }
            acc::CopyToDevice(dev.block_offsets_dev_,
                              &ctx_.block_offsets_[0],
                              sizeof(int) * ctx_.block_offsets_.size());
        }
    }

    inline void init_basis_sets(std::vector<grid_basis_set*> &set_) {
        for (auto &dev: dev_) {
            acc::SetDevice(dev.device_id_);
            dev.init_basis_sets(set_);
        }
    }

    inline void update_task_list() {
        for (auto &dev: dev_) {
            acc::SetDevice(dev.device_id_);
            dev.init_task_list(ctx_);
        }
    };
    void collocate_one_grid_level(const int level);
    void integrate_one_grid_level(const int level);

    void collocate();
    void integrate();
};
#endif
