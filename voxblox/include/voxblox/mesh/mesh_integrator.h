// The MIT License (MIT)
// Copyright (c) 2014 Matthew Klingensmith and Ivan Dryanovski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef VOXBLOX_MESH_MESH_INTEGRATOR_H_
#define VOXBLOX_MESH_MESH_INTEGRATOR_H_

#include <algorithm>
#include <list>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <glog/logging.h>
#include <Eigen/Core>

#include "voxblox/core/layer.h"
#include "voxblox/core/voxel.h"
#include "voxblox/integrator/integrator_utils.h"
#include "voxblox/interpolator/interpolator.h"
#include "voxblox/mesh/marching_cubes.h"
#include "voxblox/mesh/mesh_layer.h"
#include "voxblox/utils/meshing_utils.h"
#include "voxblox/utils/timing.h"

namespace voxblox {

struct MeshIntegratorConfig {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool use_color = true;
  float min_weight = 1e-4;

  size_t integrator_threads = std::thread::hardware_concurrency();

  inline std::string print() const {
    std::stringstream ss;
    // clang-format off
    ss << "================== Mesh Integrator Config ====================\n";
    ss << " - use_color:                 " << use_color << "\n";
    ss << " - min_weight:                " << min_weight << "\n";
    ss << " - integrator_threads:        " << integrator_threads << "\n";
    ss << "==============================================================\n";
    // clang-format on
    return ss.str();
  }
};

/**
 * Integrates a TSDF layer to incrementally update a mesh layer using marching
 * cubes.
 */
template <typename VoxelType>
class MeshIntegrator {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  void initFromSdfLayer(const Layer<VoxelType>& sdf_layer) {
    voxel_size_ = sdf_layer.voxel_size();
    block_size_ = sdf_layer.block_size();
    voxels_per_side_ = sdf_layer.voxels_per_side();

    voxel_size_inv_ = 1.0 / voxel_size_;
    block_size_inv_ = 1.0 / block_size_;
    voxels_per_side_inv_ = 1.0 / voxels_per_side_;
  }

  /**
   * Use this constructor in case you would like to modify the layer during mesh
   * extraction, i.e. modify the updated flag.
   */
  MeshIntegrator(const MeshIntegratorConfig& config,
                 Layer<VoxelType>* sdf_layer, MeshLayer* mesh_layer)
      : config_(config),
        sdf_layer_mutable_(CHECK_NOTNULL(sdf_layer)),
        sdf_layer_const_(CHECK_NOTNULL(sdf_layer)),
        mesh_layer_(CHECK_NOTNULL(mesh_layer)) {
    initFromSdfLayer(*sdf_layer);

    cube_index_offsets_ << 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
        0, 0, 1, 1, 1, 1;

    if (config_.integrator_threads == 0) {
      LOG(WARNING) << "Automatic core count failed, defaulting to 1 threads";
      config_.integrator_threads = 1;
    }
  }

  /**
   * This constructor will not allow you to modify the layer, i.e. clear the
   * updated flag.
   */
  MeshIntegrator(const MeshIntegratorConfig& config,
                 const Layer<VoxelType>& sdf_layer, MeshLayer* mesh_layer)
      : config_(config),
        sdf_layer_mutable_(nullptr),
        sdf_layer_const_(&sdf_layer),
        mesh_layer_(CHECK_NOTNULL(mesh_layer)) {
    initFromSdfLayer(sdf_layer);

    // clang-format off
    cube_index_offsets_ << 0, 1, 1, 0, 0, 1, 1, 0,
                           0, 0, 1, 1, 0, 0, 1, 1,
                           0, 0, 0, 0, 1, 1, 1, 1;
    // clang-format on

    if (config_.integrator_threads == 0) {
      LOG(WARNING) << "Automatic core count failed, defaulting to 1 threads";
      config_.integrator_threads = 1;
    }
  }

  /// Generates mesh from the tsdf layer.
  //在我们看的代码流程里 输入的两个参数都是bool值
  void generateMesh(bool only_mesh_updated_blocks, bool clear_updated_flag) {
    
    CHECK(!clear_updated_flag || (sdf_layer_mutable_ != nullptr))
        << "If you would like to modify the updated flag in the blocks, please "
        << "use the constructor that provides a non-const link to the sdf "
        << "layer!";
    //BlockIndexList = std::vector<Eigen::Vector3i>
    BlockIndexList all_tsdf_blocks;//被更新了的blokc的全局id
    //1.在我们看到代码流程里面是true
    if (only_mesh_updated_blocks) {
      //从block_map_中找到哪些block被更新了，并获取这些block的id
      sdf_layer_const_->getAllUpdatedBlocks(Update::kMesh, //in
                                            &all_tsdf_blocks);//out
    } else {
      sdf_layer_const_->getAllAllocatedBlocks(&all_tsdf_blocks);
    }

    // 2.Allocate all the mesh memory
    for (const BlockIndex& block_index : all_tsdf_blocks) {
      //判断现有地图中是否有这个block，如果有这个block直接返回这个block的指针，否则向地图插入这个block，并返回对应的指针
      mesh_layer_->allocateMeshPtrByIndex(block_index);
    }

    std::unique_ptr<ThreadSafeIndex> index_getter( new MixedThreadSafeIndex(all_tsdf_blocks.size()) );

    std::list<std::thread> integration_threads;
    //integrator_threads 默认应该是等于 1
    //3.
    for (size_t i = 0; i < config_.integrator_threads; ++i) {
      integration_threads.emplace_back( &MeshIntegrator::generateMeshBlocksFunction, //整个代码就这里调用了这个函数！
                                      this, 
                                      all_tsdf_blocks,
                                      clear_updated_flag, 
                                      index_getter.get());
    }

    for (std::thread& thread : integration_threads) {
      thread.join();
    }
  }//end function generateMesh


  void generateMeshBlocksFunction(const BlockIndexList& all_tsdf_blocks,
                                  bool clear_updated_flag,
                                  ThreadSafeIndex* index_getter) {
    DCHECK(index_getter != nullptr);
    CHECK(!clear_updated_flag || (sdf_layer_mutable_ != nullptr))
        << "If you would like to modify the updated flag in the blocks, please "
        << "use the constructor that provides a non-const link to the sdf "
        << "layer!";

    size_t list_idx;
    //遍历所有要更新的block全局id
    while (index_getter->getNextIndex(&list_idx)) {
      const BlockIndex& block_idx = all_tsdf_blocks[list_idx];
      updateMeshForBlock(block_idx);//整个代码就这里调用了这个函数， 非常重要函数！！！！！！！！
      if (clear_updated_flag) {//在我们的代码流程里面是true
        typename Block<VoxelType>::Ptr block = sdf_layer_mutable_->getBlockPtrByIndex(block_idx);//得到block对应的指针
        block->updated().reset(Update::kMesh);
      }
    }

  }//end function generateMeshBlocksFunction

  //
  void extractBlockMesh(typename Block<VoxelType>::ConstPtr block, Mesh::Ptr mesh) {
    DCHECK(block != nullptr);
    DCHECK(mesh != nullptr);

    IndexElement vps = block->voxels_per_side();//IndexElement = int
    VertexIndex next_mesh_index = 0;//size_t = VertexIndex

    VoxelIndex voxel_index;//VoxelIndex = 3*1 int 矩阵
    for (voxel_index.x() = 0; voxel_index.x() < vps - 1; ++voxel_index.x()) {
      for (voxel_index.y() = 0; voxel_index.y() < vps - 1; ++voxel_index.y()) {
        for (voxel_index.z() = 0; voxel_index.z() < vps - 1; ++voxel_index.z()) {
          Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);//计算得到这个voxel中心点在全局的三维坐标
          extractMeshInsideBlock(*block, voxel_index, coords, &next_mesh_index,
                                 mesh.get());//整个代码就这里调用了这个函数
        }
      }
    }

    // Max X plane
    // takes care of edge (x_max, y_max, z),
    // takes care of edge (x_max, y, z_max).
    voxel_index.x() = vps - 1;
    for (voxel_index.z() = 0; voxel_index.z() < vps; voxel_index.z()++) {
      for (voxel_index.y() = 0; voxel_index.y() < vps; voxel_index.y()++) {
        Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
        extractMeshOnBorder(*block, voxel_index, coords, &next_mesh_index,
                            mesh.get());
      }
    }

    // Max Y plane.
    // takes care of edge (x, y_max, z_max),
    // without corner (x_max, y_max, z_max).
    voxel_index.y() = vps - 1;
    for (voxel_index.z() = 0; voxel_index.z() < vps; voxel_index.z()++) {
      for (voxel_index.x() = 0; voxel_index.x() < vps - 1; voxel_index.x()++) {
        Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
        extractMeshOnBorder(*block, voxel_index, coords, &next_mesh_index,
                            mesh.get());
      }
    }

    // Max Z plane.
    voxel_index.z() = vps - 1;
    for (voxel_index.y() = 0; voxel_index.y() < vps - 1; voxel_index.y()++) {
      for (voxel_index.x() = 0; voxel_index.x() < vps - 1; voxel_index.x()++) {
        Point coords = block->computeCoordinatesFromVoxelIndex(voxel_index);
        extractMeshOnBorder(*block, voxel_index, coords, &next_mesh_index,
                            mesh.get());
      }
    }
  }//end function extractBlockMesh 

  virtual void updateMeshForBlock(const BlockIndex& block_index) {
    //Mesh::Ptr 本质上是共享指针
    Mesh::Ptr mesh = mesh_layer_->getMeshPtrByIndex(block_index);
    mesh->clear();
    // This block should already exist, otherwise it makes no sense to update
    // the mesh for it. ;)
    typename Block<VoxelType>::ConstPtr block = sdf_layer_const_->getBlockPtrByIndex(block_index);

    if (!block) {
      LOG(ERROR) << "Trying to mesh a non-existent block at index: "
                 << block_index.transpose();
      return;
    }
    extractBlockMesh(block, mesh);//整个代码就这里调用了这个函数，//非常重要的函数！！！！！
    // Update colors if needed.
    if (config_.use_color) {//默认这个参数是true
      updateMeshColor(*block, mesh.get());//将voxel中的颜色赋值给mesh点的颜色, 非常重要的函数！！！！！
    }

    mesh->updated = true;
  }//end function updateMeshForBlock

  void extractMeshInsideBlock(const Block<VoxelType>& block,
                              const VoxelIndex& index,
                              const Point& coords,//voxel对应的中心点坐标
                              VertexIndex* next_mesh_index, Mesh* mesh) {
    DCHECK(next_mesh_index != nullptr);
    DCHECK(mesh != nullptr);

    Eigen::Matrix<FloatingPoint, 3, 8> cube_coord_offsets =  cube_index_offsets_.cast<FloatingPoint>() * voxel_size_;
    Eigen::Matrix<FloatingPoint, 3, 8> corner_coords;//相邻voxel的中心点坐标
    Eigen::Matrix<FloatingPoint, 8, 1> corner_sdf;//每一个voxel对应的distance变量
    bool all_neighbors_observed = true;

    //遍历voxel周围的8个相邻voxel
    for (unsigned int i = 0; i < 8; ++i) {
      VoxelIndex corner_index = index + cube_index_offsets_.col(i);
      const VoxelType& voxel = block.getVoxelByVoxelIndex(corner_index);

      //将voxel的distance赋值给corner_sdf(i)
      if (!utils::getSdfIfValid(voxel, config_.min_weight,//in
                                 &(corner_sdf(i)))) //output
      {
        all_neighbors_observed = false;
        break;
      }

      corner_coords.col(i) = coords + cube_coord_offsets.col(i);
    }

    //立方体的8个点都观测到我们才进行marching cube的建立
    if (all_neighbors_observed) {
      MarchingCubes::meshCube(corner_coords, //周围voxel的中心点坐标
                              corner_sdf, //周围每个voxel对应的sdf距离
                              next_mesh_index, mesh);//very important functinon!!!!!!
    }
  }//end function extractMeshOnBorder

  void extractMeshOnBorder(const Block<VoxelType>& block,
                           const VoxelIndex& index, 
                           const Point& coords,//voxel对应的中心点坐标
                           VertexIndex* next_mesh_index, Mesh* mesh) {
    DCHECK(mesh != nullptr);

    Eigen::Matrix<FloatingPoint, 3, 8> cube_coord_offsets = cube_index_offsets_.cast<FloatingPoint>() * voxel_size_;
    Eigen::Matrix<FloatingPoint, 3, 8> corner_coords;
    Eigen::Matrix<FloatingPoint, 8, 1> corner_sdf;
    bool all_neighbors_observed = true;
    corner_coords.setZero();
    corner_sdf.setZero();

    for (unsigned int i = 0; i < 8; ++i) {
      VoxelIndex corner_index = index + cube_index_offsets_.col(i);//周围相邻voxel坐标

      if (block.isValidVoxelIndex(corner_index)) {//如果voxel在block中
        const VoxelType& voxel = block.getVoxelByVoxelIndex(corner_index);

        if (!utils::getSdfIfValid(voxel, config_.min_weight,
                                  &(corner_sdf(i)))) {
          all_neighbors_observed = false;
          break;
        }

        corner_coords.col(i) = coords + cube_coord_offsets.col(i);
      } else {
        // We have to access a different block.
        //如果voxel不在block中
        //BlockIndex = 3*1 int 矩阵
        BlockIndex block_offset = BlockIndex::Zero();//block的偏移

        for (unsigned int j = 0u; j < 3u; j++) {
          if (corner_index(j) < 0) {
            block_offset(j) = -1;
            corner_index(j) = corner_index(j) + voxels_per_side_;
          } else if (corner_index(j) >=static_cast<IndexElement>(voxels_per_side_)) {
            //IndexElement = int
            block_offset(j) = 1;
            corner_index(j) = corner_index(j) - voxels_per_side_;
          }
        }

        BlockIndex neighbor_index = block.block_index() + block_offset;
        //判断相邻的block是否存在
        if (sdf_layer_const_->hasBlock(neighbor_index)) {
          const Block<VoxelType>& neighbor_block = sdf_layer_const_->getBlockByIndex(neighbor_index);

          CHECK(neighbor_block.isValidVoxelIndex(corner_index));
          const VoxelType& voxel = neighbor_block.getVoxelByVoxelIndex(corner_index);
          //再判断这个voxel在相邻的block中是否有效
          if (!utils::getSdfIfValid(voxel, config_.min_weight,
                                    &(corner_sdf(i)))) {
            all_neighbors_observed = false;
            break;
          }

          corner_coords.col(i) = coords + cube_coord_offsets.col(i);
        } else {
          all_neighbors_observed = false;
          break;
        }
      }
    }

    if (all_neighbors_observed) {
      MarchingCubes::meshCube(corner_coords, corner_sdf, next_mesh_index, mesh);
    }
  }//end function  extractMeshOnBorder

  //将voxel中的颜色赋值给mesh点的颜色
  void updateMeshColor(const Block<VoxelType>& block, Mesh* mesh) {
    DCHECK(mesh != nullptr);

    mesh->colors.clear();
    mesh->colors.resize(mesh->indices.size());

    // Use nearest-neighbor search.
    for (size_t i = 0; i < mesh->vertices.size(); i++) {
      
      const Point& vertex = mesh->vertices[i];
      VoxelIndex voxel_index = block.computeVoxelIndexFromCoordinates(vertex);//更具点的坐标计算得到对应的voxel坐标
      if (block.isValidVoxelIndex(voxel_index)) {
        //mesh顶点->voxel坐标->对应voxel数据
        const VoxelType& voxel = block.getVoxelByVoxelIndex(voxel_index);//
        //将voxel中的颜色赋值给mesh点的颜色
        utils::getColorIfValid(voxel, config_.min_weight, &(mesh->colors[i]));//min_weight = 1e-4
      } else {
        //mesh顶点->block对应指针->voxel对应的数据
        const typename Block<VoxelType>::ConstPtr neighbor_block = sdf_layer_const_->getBlockPtrByCoordinates(vertex);
        const VoxelType& voxel = neighbor_block->getVoxelByCoordinates(vertex);
        utils::getColorIfValid(voxel, config_.min_weight, &(mesh->colors[i]));
      }
    }
  }//end function updateMeshColor

 protected:
  MeshIntegratorConfig config_;

  /**
   * Having both a const and a mutable pointer to the layer allows this
   * integrator to work both with a const layer (in case you don't want to clear
   * the updated flag) and mutable layer (in case you do want to clear the
   * updated flag).
   */
  Layer<VoxelType>* sdf_layer_mutable_;
  const Layer<VoxelType>* sdf_layer_const_;

  MeshLayer* mesh_layer_;

  // Cached map config.
  FloatingPoint voxel_size_;
  size_t voxels_per_side_;
  FloatingPoint block_size_;

  // Derived types.
  FloatingPoint voxel_size_inv_;
  FloatingPoint voxels_per_side_inv_;
  FloatingPoint block_size_inv_;

  // Cached index map.
  Eigen::Matrix<int, 3, 8> cube_index_offsets_;
};

}  // namespace voxblox

#endif  // VOXBLOX_MESH_MESH_INTEGRATOR_H_
