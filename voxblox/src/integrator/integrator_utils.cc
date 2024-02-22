#include "voxblox/integrator/integrator_utils.h"

namespace voxblox {

ThreadSafeIndex* ThreadSafeIndexFactory::get(const std::string& mode,
                                             const Pointcloud& points_C) {
  if (mode == "mixed") {
    return new MixedThreadSafeIndex(points_C.size());//构造函数只是做了变量的赋值
  } else if (mode == "sorted") {
    return new SortedThreadSafeIndex(points_C);
  } else {
    LOG(FATAL) << "Unknown integration order mode: '" << mode << "'!";
  }
  return nullptr;
}

ThreadSafeIndex::ThreadSafeIndex(size_t number_of_points)
    : atomic_idx_(0), number_of_points_(number_of_points) {}

MixedThreadSafeIndex::MixedThreadSafeIndex(size_t number_of_points)
    : ThreadSafeIndex(number_of_points),
      number_of_groups_(number_of_points / step_size_) {}//step_size_ = 1<<10

//Pointcloud 等价于std::vector<Eigen:::Vector3f>
SortedThreadSafeIndex::SortedThreadSafeIndex(const Pointcloud& points_C): ThreadSafeIndex(points_C.size()) 
{
  //indices_and_squared_norms_ 数据类型 = std::vector<std::pair<size_t, double>>
  indices_and_squared_norms_.reserve(points_C.size());
  size_t idx = 0;
  for (const Point& point_C : points_C) {
    indices_and_squared_norms_.emplace_back(idx, point_C.squaredNorm());//squaredNorm = 矩阵各个元素的平方和
    ++idx;
  }

  //然后按照点的距离进行排序
  std::sort(
      indices_and_squared_norms_.begin(), indices_and_squared_norms_.end(),
      [](const std::pair<size_t, double>& a,
         const std::pair<size_t, double>& b) { return a.second < b.second; });
}

// returns true if index is valid, false otherwise
bool ThreadSafeIndex::getNextIndex(size_t* idx) {
  DCHECK(idx != nullptr);
  size_t sequential_idx = atomic_idx_.fetch_add(1);//atomic_idx_变量线程安全 + 1

  if (sequential_idx >= number_of_points_) {
    return false;
  } else {
    *idx = getNextIndexImpl(sequential_idx);//这个函数就在下面
    return true;
  }
}

void ThreadSafeIndex::reset() { atomic_idx_.store(0); }

size_t MixedThreadSafeIndex::getNextIndexImpl(size_t sequential_idx) {
  //其中 number_of_groups_ = number_of_points / step_size_
  if (number_of_groups_ * step_size_ <= sequential_idx) {
    return sequential_idx;
  }

  const size_t group_num = sequential_idx % number_of_groups_;
  const size_t position_in_group = sequential_idx / number_of_groups_;

  return group_num * step_size_ + position_in_group;
}

size_t SortedThreadSafeIndex::getNextIndexImpl(size_t sequential_idx) {
  return indices_and_squared_norms_[sequential_idx].first;
}

//#############################################################################################################
//#############################################################################################################


// This class assumes PRE-SCALED coordinates, where one unit = one voxel size.
// The indices are also returned in this scales coordinate system, which should
// map to voxel indices.
//RayCaster实现
RayCaster::RayCaster(const Point& origin, const Point& point_G,
                     const bool is_clearing_ray,
                     const bool voxel_carving_enabled,// = true
                     const FloatingPoint max_ray_length_m,// = 5.0
                     const FloatingPoint voxel_size_inv,
                     const FloatingPoint truncation_distance,//0.1
                     const bool cast_from_origin) {// = false
                      
  const Ray unit_ray = (point_G - origin).normalized();//光线的单位方向向量

  Point ray_start, ray_end;
  if (is_clearing_ray) {
    FloatingPoint ray_length = (point_G - origin).norm();
    ray_length = std::min(std::max(ray_length - truncation_distance,
                                   static_cast<FloatingPoint>(0.0)),
                          max_ray_length_m);//光线最多5.0米
    ray_end = origin + unit_ray * ray_length;
    ray_start = voxel_carving_enabled ? origin : ray_end;
  } else {
    ray_end = point_G + unit_ray * truncation_distance;
    ray_start = voxel_carving_enabled
                    ? origin
                    : (point_G - unit_ray * truncation_distance);
  }

  const Point start_scaled = ray_start * voxel_size_inv;//计算得到起始的voxel坐标
  const Point end_scaled = ray_end * voxel_size_inv;//结束的voxel坐标

  if (cast_from_origin) {
    setupRayCaster(start_scaled, end_scaled);//very important function!!!!
  } else {
    setupRayCaster(end_scaled, start_scaled);
  }
}//end function RayCaster构造函数

RayCaster::RayCaster(const Point& start_scaled, const Point& end_scaled) {
  setupRayCaster(start_scaled, end_scaled);
}

// returns false if ray terminates at ray_index, true otherwise
//返回变量是 ray_index得到下一个voxel的索引
bool RayCaster::nextRayIndex(GlobalIndex* ray_index) {
  if (current_step_++ > ray_length_in_steps_) {
    return false;
  }

  DCHECK(ray_index != nullptr);
  *ray_index = curr_index_;

  int t_min_idx;
  t_to_next_boundary_.minCoeff(&t_min_idx);//eigen自带函数，得到矩阵中最小的数值，t_min_idx表示对应在矩阵中的索引序号
  curr_index_[t_min_idx] += ray_step_signs_[t_min_idx];
  t_to_next_boundary_[t_min_idx] += t_step_size_[t_min_idx];

  return true;
}//end function  nextRayIndex

//Point = Eigen::Vector3f
//参考 https://zhuanlan.zhihu.com/p/163277372
void RayCaster::setupRayCaster(const Point& start_scaled,
                               const Point& end_scaled) {

  if (std::isnan(start_scaled.x()) || std::isnan(start_scaled.y()) ||
      std::isnan(start_scaled.z()) || std::isnan(end_scaled.x()) ||
      std::isnan(end_scaled.y()) || std::isnan(end_scaled.z())) {
    ray_length_in_steps_ = 0;
    return;
  }

  curr_index_ = getGridIndexFromPoint<GlobalIndex>(start_scaled);
  const GlobalIndex end_index = getGridIndexFromPoint<GlobalIndex>(end_scaled);
  const GlobalIndex diff_index = end_index - curr_index_;

  current_step_ = 0;

  //这个变量没有在这个函数中用到，在其他函数中被用到了
  ray_length_in_steps_ = std::abs(diff_index.x()) + std::abs(diff_index.y()) + std::abs(diff_index.z());
  //Ray = Eigen::Vector3f
  const Ray ray_scaled = end_scaled - start_scaled;

  //signum函数获取数字的符号 大于0为1 小于0为-1
  //AnyIndex = Eigen::Vector3i
  ray_step_signs_ = AnyIndex(signum(ray_scaled.x()), signum(ray_scaled.y()), signum(ray_scaled.z()));

  const AnyIndex corrected_step(std::max(0, ray_step_signs_.x()),
                                std::max(0, ray_step_signs_.y()),
                                std::max(0, ray_step_signs_.z()));

  const Point start_scaled_shifted = start_scaled - curr_index_.cast<FloatingPoint>();

  //Ray = Eigen::Vector3f
  Ray distance_to_boundaries(corrected_step.cast<FloatingPoint>() - start_scaled_shifted);

  t_to_next_boundary_ = Ray((std::abs(ray_scaled.x()) < 0.0)? 2.0: distance_to_boundaries.x() / ray_scaled.x(),
                            (std::abs(ray_scaled.y()) < 0.0)? 2.0: distance_to_boundaries.y() / ray_scaled.y(),
                            (std::abs(ray_scaled.z()) < 0.0)? 2.0: distance_to_boundaries.z() / ray_scaled.z());

  // Distance to cross one grid cell along the ray in t.
  // Same as absolute inverse value of delta_coord.
  t_step_size_ = Ray(
      (std::abs(ray_scaled.x()) < 0.0) ? 2.0: ray_step_signs_.x() / ray_scaled.x(),
      (std::abs(ray_scaled.y()) < 0.0) ? 2.0: ray_step_signs_.y() / ray_scaled.y(),
      (std::abs(ray_scaled.z()) < 0.0) ? 2.0: ray_step_signs_.z() / ray_scaled.z());

}//end function setupRayCaster

}  // namespace voxblox
