#include "voxblox/integrator/tsdf_integrator.h"

#include <iostream>
#include <list>

namespace voxblox {

TsdfIntegratorBase::Ptr TsdfIntegratorFactory::create(
    const std::string& integrator_type_name,
    const TsdfIntegratorBase::Config& config, Layer<TsdfVoxel>* layer) {
  CHECK(!integrator_type_name.empty());

  int integrator_type = 1;
  for (const std::string& valid_integrator_type_name :
       kTsdfIntegratorTypeNames) {
    if (integrator_type_name == valid_integrator_type_name) {
      return create(static_cast<TsdfIntegratorType>(integrator_type), config,
                    layer);
    }
    ++integrator_type;
  }
  LOG(FATAL) << "Unknown TSDF integrator type: " << integrator_type_name;
  return TsdfIntegratorBase::Ptr();
}

TsdfIntegratorBase::Ptr TsdfIntegratorFactory::create(
    const TsdfIntegratorType integrator_type,
    const TsdfIntegratorBase::Config& config, Layer<TsdfVoxel>* layer) {
  CHECK_NOTNULL(layer);
  switch (integrator_type) {
    case TsdfIntegratorType::kSimple:
      return TsdfIntegratorBase::Ptr(new SimpleTsdfIntegrator(config, layer));
      break;
    case TsdfIntegratorType::kMerged:
      return TsdfIntegratorBase::Ptr(new MergedTsdfIntegrator(config, layer));
      break;
    case TsdfIntegratorType::kFast:
      return TsdfIntegratorBase::Ptr(new FastTsdfIntegrator(config, layer));
      break;
    default:
      LOG(FATAL) << "Unknown TSDF integrator type: "
                 << static_cast<int>(integrator_type);
      break;
  }
  return TsdfIntegratorBase::Ptr();
}

// Note many functions state if they are thread safe. Unless explicitly stated
// otherwise, this thread safety is based on the assumption that any pointers
// passed to the functions point to objects that are guaranteed to not be
// accessed by other threads.

TsdfIntegratorBase::TsdfIntegratorBase(const Config& config,
                                       Layer<TsdfVoxel>* layer)
    : config_(config) {
  setLayer(layer);

  if (config_.integrator_threads == 0) {
    LOG(WARNING) << "Automatic core count failed, defaulting to 1 threads";
    config_.integrator_threads = 1;
  }
  // clearing rays have no utility if voxel_carving is disabled
  if (config_.allow_clear && !config_.voxel_carving_enabled) {
    config_.allow_clear = false;
  }
}

void TsdfIntegratorBase::setLayer(Layer<TsdfVoxel>* layer) {
  CHECK_NOTNULL(layer);

  layer_ = layer;

  voxel_size_ = layer_->voxel_size();
  block_size_ = layer_->block_size();
  voxels_per_side_ = layer_->voxels_per_side();

  voxel_size_inv_ = 1.0 / voxel_size_;
  block_size_inv_ = 1.0 / block_size_;
  voxels_per_side_inv_ = 1.0 / voxels_per_side_;
}

// Will return a pointer to a voxel located at global_voxel_idx in the tsdf
// layer. Thread safe.
// Takes in the last_block_idx and last_block to prevent unneeded map lookups.
// If the block this voxel would be in has not been allocated, a block in
// temp_block_map_ is created/accessed and a voxel from this map is returned
// instead. Unlike the layer, accessing temp_block_map_ is controlled via a
// mutex allowing it to grow during integration.
// These temporary blocks can be merged into the layer later by calling
// updateLayerWithStoredBlocks()
TsdfVoxel* TsdfIntegratorBase::allocateStorageAndGetVoxelPtr(const GlobalIndex& global_voxel_idx, 
                                                              Block<TsdfVoxel>::Ptr* last_block,
                                                              BlockIndex* last_block_idx) {
  DCHECK(last_block != nullptr);
  DCHECK(last_block_idx != nullptr);

  const BlockIndex block_idx = getBlockIndexFromGlobalVoxelIndex(global_voxel_idx, voxels_per_side_inv_);

  if ((block_idx != *last_block_idx) || (*last_block == nullptr)) {
    *last_block = layer_->getBlockPtrByIndex(block_idx);
    *last_block_idx = block_idx;
  }

  // If no block at this location currently exists, we allocate a temporary
  // voxel that will be merged into the map later
  if (*last_block == nullptr) {
    // To allow temp_block_map_ to grow we can only let one thread in at once
    std::lock_guard<std::mutex> lock(temp_block_mutex_);

    typename Layer<TsdfVoxel>::BlockHashMap::iterator it = temp_block_map_.find(block_idx);
    if (it != temp_block_map_.end()) {
      *last_block = it->second;
    } else {
      auto insert_status = temp_block_map_.emplace(block_idx, 
                                                  std::make_shared<Block<TsdfVoxel>>(voxels_per_side_, 
                                                                                      voxel_size_,
                                                                                      getOriginPointFromGridIndex(block_idx, block_size_)
                                                                                    )
                                                  );

      DCHECK(insert_status.second) << "Block already exists when allocating at "
                                   << block_idx.transpose();

      *last_block = insert_status.first->second;
    }
  }

  (*last_block)->updated().set();

  const VoxelIndex local_voxel_idx = getLocalFromGlobalVoxelIndex(global_voxel_idx, voxels_per_side_);

  return &((*last_block)->getVoxelByVoxelIndex(local_voxel_idx));

}//end function  allocateStorageAndGetVoxelPtr

// NOT thread safe
//遍历temp_block_map_  向 block_map_插入数据，并清空temp_block_map_变量。
void TsdfIntegratorBase::updateLayerWithStoredBlocks() {
  BlockIndex last_block_idx;
  Block<TsdfVoxel>::Ptr block = nullptr;
  
  //temp_block_map_ = std::unordered_map<Eigen::Vector3i,  Block<TsdfVoxel>::Ptr>
  for (const std::pair<const BlockIndex, Block<TsdfVoxel>::Ptr>& temp_block_pair : temp_block_map_) {
    layer_->insertBlock(temp_block_pair);//向block_map_中添加数据
  }

  temp_block_map_.clear();
}

// Updates tsdf_voxel. Thread safe.
void TsdfIntegratorBase::updateTsdfVoxel(const Point& origin,//载体在世界坐标系下的坐标
                                         const Point& point_G,//点在世界坐标系下的坐标
                                         const GlobalIndex& global_voxel_idx,// GlobalIndex = Eigen::Matrix<int64_t, 3,1 >
                                         const Color& color, const float weight,
                                         TsdfVoxel* tsdf_voxel) {
  DCHECK(tsdf_voxel != nullptr);

  const Point voxel_center = getCenterPointFromGridIndex(global_voxel_idx, voxel_size_);//根据voxel id 得到这个voxel在世界坐标系下的中心点坐标

  const float sdf = computeDistance(origin, point_G, voxel_center);

  float updated_weight = weight;
  // Compute updated weight in case we use weight dropoff. It's easier here
  // that in getVoxelWeight as here we have the actual SDF for the voxel
  // already computed.
  const FloatingPoint dropoff_epsilon = voxel_size_;
  //use_weight_dropoff = 默认等于true;
  if (config_.use_weight_dropoff && sdf < -dropoff_epsilon) {
    //default_truncation_distance = 默认0.1
    updated_weight = weight * (config_.default_truncation_distance + sdf) /(config_.default_truncation_distance - dropoff_epsilon);
    updated_weight = std::max(updated_weight, 0.0f);
  }

  // Compute the updated weight in case we compensate for sparsity. By
  // multiplicating the weight of occupied areas (|sdf| < truncation distance)
  // by a factor, we prevent to easily fade out these areas with the free
  // space parts of other rays which pass through the corresponding voxels.
  // This can be useful for creating a TSDF map from sparse sensor data (e.g.
  // visual features from a SLAM system). By default, this option is disabled.
  //默认等于false 不进入这个条件
  if (config_.use_sparsity_compensation_factor) {
    if (std::abs(sdf) < config_.default_truncation_distance) {
      updated_weight *= config_.sparsity_compensation_factor;
    }
  }

  // Lookup the mutex that is responsible for this voxel and lock it
  std::lock_guard<std::mutex> lock(mutexes_.get(global_voxel_idx));

  const float new_weight = tsdf_voxel->weight + updated_weight;

  // it is possible to have weights very close to zero, due to the limited
  // precision of floating points dividing by this small value can cause nans
  if (new_weight < kFloatEpsilon) {//kFloatEpsilon = 1e-6
    return;
  }

  const float new_sdf = (sdf * updated_weight + tsdf_voxel->distance * tsdf_voxel->weight) / new_weight;

  // color blending is expensive only do it close to the surface
  if (std::abs(sdf) < config_.default_truncation_distance) {
    tsdf_voxel->color = Color::blendTwoColors( tsdf_voxel->color, tsdf_voxel->weight, 
                                               color, updated_weight );
  }
  tsdf_voxel->distance = (new_sdf > 0.0) ? std::min(config_.default_truncation_distance, new_sdf) : std::max(-config_.default_truncation_distance, new_sdf);
  //config_.max_weight = 10000.0
  tsdf_voxel->weight = std::min(config_.max_weight, new_weight);


}//end function updateTsdfVoxel



// Thread safe.
// Figure out whether the voxel is behind or in front of the surface.
// To do this, project the voxel_center onto the ray from origin to point G.
// Then check if the the magnitude of the vector is smaller or greater than
// the original distance...
float TsdfIntegratorBase::computeDistance(const Point& origin,
                                          const Point& point_G,
                                          const Point& voxel_center) const {
  const Point v_voxel_origin = voxel_center - origin;
  const Point v_point_origin = point_G - origin;

  const FloatingPoint dist_G = v_point_origin.norm();
  // projection of a (v_voxel_origin) onto b (v_point_origin)
  const FloatingPoint dist_G_V = v_voxel_origin.dot(v_point_origin) / dist_G;

  const float sdf = static_cast<float>(dist_G - dist_G_V);
  return sdf;
}

// Thread safe.
//3d点越远则权重越小
float TsdfIntegratorBase::getVoxelWeight(const Point& point_C) const {
  if (config_.use_const_weight) {//默认等于 false
    return 1.0f;
  }
  const FloatingPoint dist_z = std::abs(point_C.z());
  if (dist_z > kEpsilon) {//1e-6 = kEpsilon
    return 1.0f / (dist_z * dist_z);
  }
  return 0.0f;
}

void SimpleTsdfIntegrator::integratePointCloud(const Transformation& T_G_C,
                                               const Pointcloud& points_C,
                                               const Colors& colors,
                                               const bool freespace_points) {
  timing::Timer integrate_timer("integrate/simple");
  CHECK_EQ(points_C.size(), colors.size());

  std::unique_ptr<ThreadSafeIndex> index_getter(
      ThreadSafeIndexFactory::get(config_.integration_order_mode, points_C));

  std::list<std::thread> integration_threads;
  for (size_t i = 0; i < config_.integrator_threads; ++i) {
    integration_threads.emplace_back(&SimpleTsdfIntegrator::integrateFunction,
                                     this, T_G_C, points_C, colors,
                                     freespace_points, index_getter.get());
  }

  for (std::thread& thread : integration_threads) {
    thread.join();
  }
  integrate_timer.Stop();

  timing::Timer insertion_timer("inserting_missed_blocks");
  updateLayerWithStoredBlocks();
  insertion_timer.Stop();
}

void SimpleTsdfIntegrator::integrateFunction(const Transformation& T_G_C,
                                             const Pointcloud& points_C,
                                             const Colors& colors,
                                             const bool freespace_points,
                                             ThreadSafeIndex* index_getter) {
  DCHECK(index_getter != nullptr);

  size_t point_idx;
  while (index_getter->getNextIndex(&point_idx)) {
    const Point& point_C = points_C[point_idx];
    const Color& color = colors[point_idx];
    bool is_clearing;
    if (!isPointValid(point_C, freespace_points, &is_clearing)) {
      continue;
    }

    const Point origin = T_G_C.getPosition();
    const Point point_G = T_G_C * point_C;

    RayCaster ray_caster(origin, point_G, is_clearing,
                         config_.voxel_carving_enabled,
                         config_.max_ray_length_m, voxel_size_inv_,
                         config_.default_truncation_distance);

    Block<TsdfVoxel>::Ptr block = nullptr;
    BlockIndex block_idx;
    GlobalIndex global_voxel_idx;
    while (ray_caster.nextRayIndex(&global_voxel_idx)) {
      TsdfVoxel* voxel =
          allocateStorageAndGetVoxelPtr(global_voxel_idx, &block, &block_idx);

      const float weight = getVoxelWeight(point_C);

      updateTsdfVoxel(origin, point_G, global_voxel_idx, color, weight, voxel);
    }
  }
}

void MergedTsdfIntegrator::integratePointCloud(const Transformation& T_G_C,
                                               const Pointcloud& points_C,
                                               const Colors& colors,
                                               const bool freespace_points) {
  timing::Timer integrate_timer("integrate/merged");
  CHECK_EQ(points_C.size(), colors.size());

  // Pre-compute a list of unique voxels to end on.
  // Create a hashmap: VOXEL INDEX -> index in original cloud.
  LongIndexHashMapType<AlignedVector<size_t>>::type voxel_map;
  // This is a hash map (same as above) to all the indices that need to be
  // cleared.
  LongIndexHashMapType<AlignedVector<size_t>>::type clear_map;

  std::unique_ptr<ThreadSafeIndex> index_getter(
      ThreadSafeIndexFactory::get(config_.integration_order_mode, points_C));

  bundleRays(T_G_C, points_C, freespace_points, index_getter.get(), &voxel_map,
             &clear_map);

  integrateRays(T_G_C, points_C, colors, config_.enable_anti_grazing, false,
                voxel_map, clear_map);

  timing::Timer clear_timer("integrate/clear");

  integrateRays(T_G_C, points_C, colors, config_.enable_anti_grazing, true,
                voxel_map, clear_map);

  clear_timer.Stop();

  integrate_timer.Stop();
}

void MergedTsdfIntegrator::bundleRays(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const bool freespace_points, ThreadSafeIndex* index_getter,
    LongIndexHashMapType<AlignedVector<size_t>>::type* voxel_map,
    LongIndexHashMapType<AlignedVector<size_t>>::type* clear_map) {
  DCHECK(voxel_map != nullptr);
  DCHECK(clear_map != nullptr);

  size_t point_idx;
  while (index_getter->getNextIndex(&point_idx)) {
    const Point& point_C = points_C[point_idx];
    bool is_clearing;
    if (!isPointValid(point_C, freespace_points, &is_clearing)) {
      continue;
    }

    const Point point_G = T_G_C * point_C;

    GlobalIndex voxel_index =
        getGridIndexFromPoint<GlobalIndex>(point_G, voxel_size_inv_);

    if (is_clearing) {
      (*clear_map)[voxel_index].push_back(point_idx);
    } else {
      (*voxel_map)[voxel_index].push_back(point_idx);
    }
  }

  VLOG(3) << "Went from " << points_C.size() << " points to "
          << voxel_map->size() << " raycasts  and " << clear_map->size()
          << " clear rays.";
}

void MergedTsdfIntegrator::integrateVoxel(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, bool enable_anti_grazing, bool clearing_ray,
    const std::pair<GlobalIndex, AlignedVector<size_t>>& kv,
    const LongIndexHashMapType<AlignedVector<size_t>>::type& voxel_map) {
  if (kv.second.empty()) {
    return;
  }

  const Point& origin = T_G_C.getPosition();
  Color merged_color;
  Point merged_point_C = Point::Zero();
  FloatingPoint merged_weight = 0.0;

  for (const size_t pt_idx : kv.second) {
    const Point& point_C = points_C[pt_idx];
    const Color& color = colors[pt_idx];

    const float point_weight = getVoxelWeight(point_C);
    if (point_weight < kEpsilon) {
      continue;
    }
    merged_point_C = (merged_point_C * merged_weight + point_C * point_weight) /
                     (merged_weight + point_weight);
    merged_color =
        Color::blendTwoColors(merged_color, merged_weight, color, point_weight);
    merged_weight += point_weight;

    // only take first point when clearing
    if (clearing_ray) {
      break;
    }
  }

  const Point merged_point_G = T_G_C * merged_point_C;

  RayCaster ray_caster(origin, merged_point_G, clearing_ray,
                       config_.voxel_carving_enabled, config_.max_ray_length_m,
                       voxel_size_inv_, config_.default_truncation_distance);

  GlobalIndex global_voxel_idx;
  while (ray_caster.nextRayIndex(&global_voxel_idx)) {
    if (enable_anti_grazing) {
      // Check if this one is already the the block hash map for this
      // insertion. Skip this to avoid grazing.
      if ((clearing_ray || global_voxel_idx != kv.first) &&
          voxel_map.find(global_voxel_idx) != voxel_map.end()) {
        continue;
      }
    }

    Block<TsdfVoxel>::Ptr block = nullptr;
    BlockIndex block_idx;
    TsdfVoxel* voxel =
        allocateStorageAndGetVoxelPtr(global_voxel_idx, &block, &block_idx);

    updateTsdfVoxel(origin, merged_point_G, global_voxel_idx, merged_color,
                    merged_weight, voxel);
  }
}

void MergedTsdfIntegrator::integrateVoxels(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, bool enable_anti_grazing, bool clearing_ray,
    const LongIndexHashMapType<AlignedVector<size_t>>::type& voxel_map,
    const LongIndexHashMapType<AlignedVector<size_t>>::type& clear_map,
    size_t thread_idx) {
  LongIndexHashMapType<AlignedVector<size_t>>::type::const_iterator it;
  size_t map_size;
  if (clearing_ray) {
    it = clear_map.begin();
    map_size = clear_map.size();
  } else {
    it = voxel_map.begin();
    map_size = voxel_map.size();
  }

  for (size_t i = 0; i < map_size; ++i) {
    if (((i + thread_idx + 1) % config_.integrator_threads) == 0) {
      integrateVoxel(T_G_C, points_C, colors, enable_anti_grazing, clearing_ray,
                     *it, voxel_map);
    }
    ++it;
  }
}

void MergedTsdfIntegrator::integrateRays(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, bool enable_anti_grazing, bool clearing_ray,
    const LongIndexHashMapType<AlignedVector<size_t>>::type& voxel_map,
    const LongIndexHashMapType<AlignedVector<size_t>>::type& clear_map) {
  // if only 1 thread just do function call, otherwise spawn threads
  if (config_.integrator_threads == 1) {
    constexpr size_t thread_idx = 0;
    integrateVoxels(T_G_C, points_C, colors, enable_anti_grazing, clearing_ray,
                    voxel_map, clear_map, thread_idx);
  } else {
    std::list<std::thread> integration_threads;
    for (size_t i = 0; i < config_.integrator_threads; ++i) {
      integration_threads.emplace_back(
          &MergedTsdfIntegrator::integrateVoxels, this, T_G_C, points_C, colors,
          enable_anti_grazing, clearing_ray, voxel_map, clear_map, i);
    }

    for (std::thread& thread : integration_threads) {
      thread.join();
    }
  }

  timing::Timer insertion_timer("inserting_missed_blocks");
  updateLayerWithStoredBlocks();

  insertion_timer.Stop();
}

//
void FastTsdfIntegrator::integrateFunction(const Transformation& T_G_C,
                                           const Pointcloud& points_C,
                                           const Colors& colors,
                                           const bool freespace_points,//默认等于false
                                           ThreadSafeIndex* index_getter) {
  DCHECK(index_getter != nullptr);

  size_t point_idx;
  //1.将点云中的点逐个取出来
  while (index_getter->getNextIndex(&point_idx) && //是线程安全的
        //后面这个时间判断不管，基本上没有生效
         ( std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - integration_start_time_).count() < config_.max_integration_time_s * 1000000 )
        )
  {
    const Point& point_C = points_C[point_idx];
    const Color& color = colors[point_idx];
    bool is_clearing;
    //2.根据距离筛选点
    if (!isPointValid(point_C, freespace_points, &is_clearing)) {
      continue;
    }

    const Point origin = T_G_C.getPosition();//载体原点
    const Point point_G = T_G_C * point_C;//3d点在世界坐标系下的坐标
    // Checks to see if another ray in this scan has already started 'close'
    // to this location. If it has then we skip ray casting this point. We
    // measure if a start location is 'close' to another points by inserting
    // the point into a set of voxels. This voxel set has a resolution
    // start_voxel_subsampling_factor times higher then the voxel size.

    //GlobalIndex  = matrix<int64_t,3,1>
    //3.获取global的id
    GlobalIndex global_voxel_idx;
    //start_voxel_subsampling_factor = 2.0
    global_voxel_idx = getGridIndexFromPoint<GlobalIndex>(point_G, 
                                                          config_.start_voxel_subsampling_factor * voxel_size_inv_);
    //start_voxel_approx_set_这个变量仅是用来判断地图中的voxel是否重复，如果两个点对应相同的                                                
    if (!start_voxel_approx_set_.replaceHash(global_voxel_idx)) {//搜索 inline bool replaceHash(const IndexType& index) {
      continue;
    }


    //4.构建ray_caster, 
    constexpr bool cast_from_origin = false;
    //搜索 RayCaster实现
    RayCaster ray_caster(origin, point_G, is_clearing,
                         config_.voxel_carving_enabled,//voxel_carving_enabled = true
                         config_.max_ray_length_m, voxel_size_inv_,
                         config_.default_truncation_distance, cast_from_origin);//default_truncation_distance = 0.1

    int64_t consecutive_ray_collisions = 0;

    Block<TsdfVoxel>::Ptr block = nullptr;
    BlockIndex block_idx;
    while (ray_caster.nextRayIndex(&global_voxel_idx)) {
      // Check if the current voxel has been seen by any ray cast this scan.
      // If it has increment the consecutive_ray_collisions counter, otherwise
      // reset it. If the counter reaches a threshold we stop casting as the
      // ray is deemed to be contributing too little new information.
      if (!voxel_observed_approx_set_.replaceHash(global_voxel_idx)) {
        ++consecutive_ray_collisions;
      } else {
        consecutive_ray_collisions = 0;
      }
      if (consecutive_ray_collisions > config_.max_consecutive_ray_collisions) {
        break;
      }

      TsdfVoxel* voxel = allocateStorageAndGetVoxelPtr(global_voxel_idx, &block, &block_idx);//非常重要的函数！！修改了 temp_block_map_变量

      const float weight = getVoxelWeight(point_C);//3d点越远则权重越小,小函数

      //本文件搜索 TsdfIntegratorBase::updateTsdfVoxel
      updateTsdfVoxel(origin, point_G, global_voxel_idx, color, weight, voxel); //非常重要的函数！！！！！！!!!!!
    }
  }//end while
}//end function  integrateFunction


//使用多个线程处理点云数据
void FastTsdfIntegrator::integratePointCloud(const Transformation& T_G_C,//点云对应的位姿
                                             const Pointcloud& points_C,//点云的xyz坐标
                                             const Colors& colors,//点云对应的颜色
                                             const bool freespace_points) {//默认等于false
  timing::Timer integrate_timer("integrate/fast");
  CHECK_EQ(points_C.size(), colors.size());

  integration_start_time_ = std::chrono::steady_clock::now();//整个代码就这里被赋值了！！！！

  //clear_checks_every_n_frames  = 默认 1
  static int64_t reset_counter = 0;
  if ((++reset_counter) >= config_.clear_checks_every_n_frames) {
    reset_counter = 0;
    start_voxel_approx_set_.resetApproxSet();
    voxel_observed_approx_set_.resetApproxSet();
  }

  //integration_order_mode = mixed
  std::unique_ptr<ThreadSafeIndex> index_getter( ThreadSafeIndexFactory::get(config_.integration_order_mode, points_C) );//只是做了简单的变量赋值

  std::list<std::thread> integration_threads;
  //config_.integrator_threads应该默认等于1
  for (size_t i = 0; i < config_.integrator_threads; ++i) {
    //1.
    integration_threads.emplace_back(&FastTsdfIntegrator::integrateFunction,//非常重要的函数！！！！！！！！！！！！！！！！！！
                                     this, 
                                     T_G_C, 
                                     points_C, 
                                     colors,
                                     freespace_points, 
                                     index_getter.get());
  }

  for (std::thread& thread : integration_threads) {
    thread.join();
  }

  integrate_timer.Stop();

  timing::Timer insertion_timer("inserting_missed_blocks");
  //2.遍历temp_block_map_  向 block_map_插入数据，并清空temp_block_map_变量。
  //本文件搜索 TsdfIntegratorBase::updateLayerWithStoredBlocks() {
  updateLayerWithStoredBlocks();

  insertion_timer.Stop();
}//end function integratePointCloud

std::string TsdfIntegratorBase::Config::print() const {
  std::stringstream ss;
  // clang-format off
  ss << "================== TSDF Integrator Config ====================\n";
  ss << " General: \n";
  ss << " - default_truncation_distance:               " << default_truncation_distance << "\n";
  ss << " - max_weight:                                " << max_weight << "\n";
  ss << " - voxel_carving_enabled:                     " << voxel_carving_enabled << "\n";
  ss << " - min_ray_length_m:                          " << min_ray_length_m << "\n";
  ss << " - max_ray_length_m:                          " << max_ray_length_m << "\n";
  ss << " - use_const_weight:                          " << use_const_weight << "\n";
  ss << " - allow_clear:                               " << allow_clear << "\n";
  ss << " - use_weight_dropoff:                        " << use_weight_dropoff << "\n";
  ss << " - use_sparsity_compensation_factor:          " << use_sparsity_compensation_factor << "\n";
  ss << " - sparsity_compensation_factor:              "  << sparsity_compensation_factor << "\n";
  ss << " - integrator_threads:                        " << integrator_threads << "\n";
  ss << " MergedTsdfIntegrator: \n";
  ss << " - enable_anti_grazing:                       " << enable_anti_grazing << "\n";
  ss << " FastTsdfIntegrator: \n";
  ss << " - start_voxel_subsampling_factor:            " << start_voxel_subsampling_factor << "\n";
  ss << " - max_consecutive_ray_collisions:            " << max_consecutive_ray_collisions << "\n";
  ss << " - clear_checks_every_n_frames:               " << clear_checks_every_n_frames << "\n";
  ss << " - max_integration_time_s:                    " << max_integration_time_s << "\n";
  ss << "==============================================================\n";
  // clang-format on
  return ss.str();
}

}  // namespace voxblox
