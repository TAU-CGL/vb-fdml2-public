#pragma once

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <fstream>
#include <sstream>

#include <omp.h>
#include <fmt/core.h>
#include <glm/glm.hpp>

#include <CGAL/basic.h>
#include <CGAL/intersections.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Arr_non_caching_segment_traits_2.h>
#include <CGAL/Arrangement_2.h>
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::FT                                          Number_type;
typedef CGAL::Arr_non_caching_segment_traits_2<Kernel>      Traits;
typedef Traits::Ray_2                                       Ray_2;
typedef Traits::Point_2                                     Point_2;
typedef Traits::X_monotone_curve_2                          Segment_2;
typedef CGAL::Arrangement_2<Traits>                         Arrangement_2;

#define OMP_INIT() omp_set_num_threads(omp_get_max_threads())

#define GENERATE_ARRANGEMENT(scene) Arrangement_2 arr;\
        std::vector<Segment_2> segments;\
        for (auto segment : scene) segments.push_back(Segment_2(Point_2(segment.x1, segment.y1), Point_2(segment.x2, segment.y2)));\
        CGAL::insert(arr, segments.begin(), segments.end());
#define COMPUTE_DISTANCE(arr, x, y, theta) \
        float sin_theta = sin(theta);\
        float cos_theta = cos(theta);\
        Point_2 p(x, y);\
        Point_2 direction(x + cos_theta, y + sin_theta);\
        Ray_2 ray(Point_2(x, y), direction);\
        float dist = INFINITY;\
        for (auto it = arr.edges_begin(); it != arr.edges_end(); ++it) {\
            if (auto isect = CGAL::intersection(ray, it->curve())) {\
                if (const Point_2* ipoint = boost::get<Point_2>(&*isect)) {\
                    dist = sqrt(CGAL::to_double(CGAL::squared_distance(p, *ipoint)));\
                    break;\
                }\
            }\
        }

namespace vbfdml2 {
    
    struct Segment;
    struct BoxExtent;
    struct VoxelCloud;
    struct Prediction;
    struct Measurement;
    struct DynamicObstacle;
    
    // A 2D planar segment (x1,y1) <--> (x2,y2)
    struct Segment {
    public:
        Segment(float x1, float y1, float x2, float y2);
        float x1, y1;
        float x2, y2;

        std::string ToString() const;
    };

    // Represents the box extent for which we compute the marching voxels algorithm
    struct BoxExtent {
    public:
        BoxExtent(float width, float height, float depth, float x0, float y0, float theta0);
        float width, height, depth;
        float x0, y0, theta0;

        std::string ToString() const;
    };
    
    // A (n x n x n) voxel cloud 
    struct VoxelCloud
    {
    public:
        void Init(int n);
        void Delete();

        // float* buffer = nullptr;
        std::vector<float> buffer;
        int n;

        std::string ToString() const;
        std::vector<float> ToList() const;
        void ExportOBJ(const char* path, BoxExtent& extent);
    };

    // A localization (SE(2)) prediction
    struct Prediction
    {
    public:
        float x, y, theta;
        int cnt;

        std::string ToString() const;
    };

    struct DynamicObstacle
    {
    public:
        DynamicObstacle(float x, float y, float radius, std::vector<std::pair<float, float>> path);
        float x, y, radius;
        // list of 2d coordinates of the objects path
        std::vector<std::pair<float, float>> path;

        std::string ToString() const;
    };


    // An SE(2) rigid body transformation + measured distance
    struct Measurement
    {
    public:
        Measurement(float dx, float dy, float dtheta, float dist, bool d_obstacle = false);

        float dx, dy, dtheta;
        float dist;
        bool d_obstacle;

        std::string ToString() const;
    };

    // Computes the distance function h(x,y,theta)
    // This can be used as pre-processing and run only once per scene
    // Also output a voxel cloud mask which is 1 for each voxel that is inside the room
    void PreprocessScene(VoxelCloud& output, VoxelCloud& mask, BoxExtent& extent, std::vector<Segment> scene, bool bUseGPU = true);

    // Given a preprocessed scene, compute the preimage via the marching voxels algorithm, after
    // some pre-determined planar motion (dx, dy, dtheta) and recieving a measurment (d)
    void MarchingVoxels(VoxelCloud& output, BoxExtent& extent, VoxelCloud& preprocess, float dx, float dy, float dtheta, float d, bool bUseGPU = true);

    // Apply 3D convolution to a voxel cloud (the heuristic described in the paper for achieveing better completeness)
    void Conv3D(VoxelCloud& output, VoxelCloud& m, bool bUseGPU = true);

    // Intersect voxel clouds 
    void DoIntersect(VoxelCloud& output, VoxelCloud& m1, VoxelCloud& m2, bool bUseGPU = true);

    // Outputs predictions of center of mass of each component in the voxel cloud
    void Predict(VoxelCloud& vc, BoxExtent& extent, std::vector<Prediction>& predictions);

    // Run the improved FDML method (robust to dynamic obstacles)
    void RunImprovedVBFDML(std::vector<Segment> scene, BoxExtent& extent, std::vector<Measurement> measurements, int n);

    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    inline Segment::Segment(float x1, float y1, float x2, float y2) : x1(x1), y1(y1), x2(x2), y2(y2) {}
    inline std::string Segment::ToString() const {
        return fmt::format("SEGMENT\n\t({},{}) <--> ({},{})", x1, y1, x2, y2);
    }
    inline BoxExtent::BoxExtent(float width, float height, float depth, float x0, float y0, float theta0) : width(width), height(height), depth(depth), x0(x0), y0(y0), theta0(theta0) {}
    inline std::string BoxExtent::ToString() const
    {
        return std::string("BOX_EXTENT\n\t") +
            std::string("DIMENSIONS W = ") + std::to_string(width) +
            std::string(", H = ") + std::to_string(height) +
            std::string(", D = ") + std::to_string(depth) +
            std::string("\n\tOFFSET (") +
            std::to_string(x0) + std::string(",") + std::to_string(y0) + std::string(",") + std::to_string(theta0) + std::string(")");
    }
    
    inline void VoxelCloud::Init(int n)
    {
        // this->buffer = new float[n * n * n];
        this->buffer = std::vector<float>(n * n * n, 0.0f);
        this->n = n;
    }

    inline void VoxelCloud::Delete()
    {
        // if (this->buffer)
        //     delete[] this->buffer;
    }

    inline std::vector<float> VoxelCloud::ToList() const
    {
        std::vector<float> res;
        // if (this->buffer)
        if (this->buffer.size() > 0)
            for(int i = 0; i < n * n * n; i++)
                res.push_back(this->buffer[i]);
        return res;
    }

    inline std::string VoxelCloud::ToString() const
    {
        std::string sn = std::to_string(n);
        std::string cm = std::string(" x ");
        return std::string("VOXEL_CLOUD (") + sn + cm + sn + cm + sn + std::string(")");
    }

    inline void VoxelCloud::ExportOBJ(const char *path, BoxExtent &extent)
    {
        std::ofstream fp(path);

        int cnt = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                {
                    if (buffer[i + j * n + k * n * n] <= 0)
                        continue;
                    float x = (-0.5f + (float)i / (float)(n - 1)) * extent.width + extent.x0;
                    float y = (-0.5f + (float)j / (float)(n - 1)) * extent.height + extent.y0;
                    float theta = (-0.5f + (float)k / (float)(n - 1)) * extent.depth + extent.theta0;
                    fp << "v " << x << " " << y << " " << theta << std::endl;
                    cnt++;
                }

        // Export each point as a separate (degenerate) face
        // std::cout << "Num points: " << cnt << std::endl;
        for (int f = 0; f < cnt; f++)
            fp << "f " << (f + 1) << " " << (f + 1) << " " << (f + 1) << std::endl;
        fp.close();
    }

    inline std::string Prediction::ToString() const
    {
        return std::string("PREDICTION\n\t(") + 
            std::to_string(x) + std::string(",") + std::to_string(y) + std::string(") theta = ") + std::to_string(theta) +
            std::string(" [n=") + std::to_string(cnt) + std::string("]");
    }

    inline std::string DynamicObstacle::ToString() const
    {
        return std::string("DYNAMIC OBSTACLE\n\t(") + 
            std::to_string(x) + std::string(",") + std::to_string(y) + std::string(") radius = ") + std::to_string(radius);
    }

    inline DynamicObstacle::DynamicObstacle(float x, float y, float radius, std::vector<std::pair<float, float>> path) : x(x), y(y), radius(radius), path(path)
    {
    }

    inline Measurement::Measurement(float dx, float dy, float dtheta, float dist, bool d_obstacle) : dx(dx), dy(dy), dtheta(dtheta), dist(dist), d_obstacle(d_obstacle)
    {
    }

    inline std::string Measurement::ToString() const
    {
        return std::string("MEASUREMENT\n\t(") + 
            std::to_string(dx) + std::string(",") + std::to_string(dy) + std::string(") theta = ") + std::to_string(dtheta) +
            std::string(" [dist=") + std::to_string(dist) + std::string("]");
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void PreprocessScene(VoxelCloud& output, VoxelCloud& mask, BoxExtent& extent, std::vector<Segment> scene, bool bUseGPU) {
        OMP_INIT();

        GENERATE_ARRANGEMENT(scene);

        #pragma omp parallel for collapse(3)
        for (int i = 0; i < output.n; i++)
        for (int j = 0; j < output.n; j++)
        for (int k = 0; k < output.n; k++) {
            float x = (-0.5f + (float)i / (float)(output.n - 1)) * extent.width + extent.x0;
            float y = (-0.5f + (float)j / (float)(output.n - 1)) * extent.height + extent.y0;
            float theta = (-0.5f + (float)k / (float)(output.n - 1)) * extent.depth + extent.theta0;

            // Compute the distance to the scene ("imp_f" in the legacy version)
            COMPUTE_DISTANCE(arr, x, y, theta);
            
            output.buffer[i + j * output.n + k * output.n * output.n] = dist;
            mask.buffer[i + j * output.n + k * output.n * output.n] = 1.0;
        }
    }

    void MarchingVoxels(VoxelCloud& output, BoxExtent& extent, VoxelCloud& preprocess, float dx, float dy, float dtheta, float d, bool bUseGPU) {
        OMP_INIT();

        #pragma omp parallel for collapse(3)
        for (int i = 0; i < output.n; i++)
        for (int j = 0; j < output.n; j++)
        for (int k = 0; k < output.n; k++) {
            float x = (-0.5f + (float)i / (float)(output.n - 1)) * extent.width + extent.x0;
            float y = (-0.5f + (float)j / (float)(output.n - 1)) * extent.height + extent.y0;
            float theta = (-0.5f + (float)k / (float)(output.n - 1)) * extent.depth + extent.theta0;

            // First compose the delta transformation
            float x_ = x + dy * cos(theta + dtheta) - dx * sin(theta + dtheta);
            float y_ = y + dy * sin(theta + dtheta) + dx * cos(theta + dtheta);
            float theta_ = theta + dtheta;
            if (theta_ > M_PI) theta_ -= 2 * M_PI;
            if (theta_ < -M_PI) theta_ += 2 * M_PI;

            // Now get the lookup index of the new location (and cutoff invalid indexes)
            int i_ = (int)((float)(output.n-1) * (0.5f + (float)(x_ - extent.x0) / (float)extent.width));
            int j_ = (int)((float)(output.n-1) * (0.5f + (float)(y_ - extent.y0) / (float)extent.height));
            int k_ = (int)((float)(output.n-1) * (0.5f + (float)(theta_ - extent.theta0) / (float)extent.depth));
            if (i_ < 0 || i_ >= output.n || j_ < 0 || j_ >= output.n || k_ < 0 || k_ >= output.n) continue;

            // Query all grid neighbors for marching voxel method
            char pos = 0, neg = 0;
            for (int i__ = -1; i__ <= 1; i__++)
            for (int j__ = -1; j__ <= 1; j__++)
            for (int k__ = -1; k__ <= 1; k__++) {
                if ((i_+i__) < 0 || (i_+i__) >= output.n || (j_+j__) < 0 || (j_+j__) >= output.n || (k_+k__) < 0 || (k_+k__) >= output.n) continue;
                float val = preprocess.buffer[(i_+i__) + (j_+j__) * output.n + (k_+k__) * output.n * output.n] - d;
                if (val == INFINITY) continue;
                if (val >= 0.f) pos = 1;
                if (val <= 0.f) neg = 1;
                if (pos && neg) break;
            }

            output.buffer[i + j * output.n + k * output.n * output.n] = pos && neg;
        }
    }

    void Conv3D(VoxelCloud& output, VoxelCloud& m, bool bUseGPU) {
        OMP_INIT();

        #pragma omp parallel for collapse(3)
        for (int i_ = 0; i_ < output.n; i_++)
        for (int j_ = 0; j_ < output.n; j_++)
        for (int k_ = 0; k_ < output.n; k_++) {
            float val = 0.0f;
            for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
            for (int k = -1; k <= 1; k++) {
                if ((i_+i) < 0 || (i_+i) >= output.n || (j_+j) < 0 || (j_+j) >= output.n || (k_+k) < 0 || (k_+k) >= output.n) continue;
                val += m.buffer[(i_ + i) + (j_ + j) * output.n + (k_ + k) * output.n * output.n];
            }
            output.buffer[i_ + j_ * output.n + k_ * output.n * output.n] = val;
        }
    }

    void DoIntersect(VoxelCloud& output, VoxelCloud& m1, VoxelCloud& m2, bool bUseGPU) {
        OMP_INIT();

        #pragma omp parallel for
        for (int idx = 0; idx < output.n * output.n * output.n; idx++) {
            output.buffer[idx] = (m1.buffer[idx] * m2.buffer[idx] > 0.5f) ? 1.0f : 0.0f;
        }
    }

    inline void _PredictDFS(VoxelCloud& vc, VoxelCloud& visited, BoxExtent& extent, int i, int j, int k, Prediction& prediction, int depth)
    {
        // if (depth > 800) return;

        for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dz = -1; dz <= 1; dz++)
        {
            if (((i+dx) < 0) || ((i+dx) >= vc.n)) continue;
            if (((j+dy) < 0) || ((j+dy) >= vc.n)) continue;
            if (((k+dz) < 0) || ((k+dz) >= vc.n)) continue;

            int new_idx = (i+dx) + (j+dy) * vc.n + (k+dz) * vc.n * vc.n;

            if (vc.buffer[new_idx] < 0.5f) continue;
            if (visited.buffer[new_idx] > 999.f) continue;

            prediction.x += (-0.5f + (float)(i+dx)/(float)(vc.n-1)) * extent.width + extent.x0;
            prediction.y += (-0.5f + (float)(j+dy)/(float)(vc.n-1)) * extent.height + extent.y0;
            prediction.theta += (-0.5f + (float)(k+dz)/(float)(vc.n-1)) * extent.depth + extent.theta0;
            prediction.cnt += 1;

            visited.buffer[new_idx] = 1000.f;
            _PredictDFS(vc, visited, extent, i+dx, j+dy, k+dz, prediction, depth + 1);
        }
    }

    inline void Predict(VoxelCloud& vc, BoxExtent& extent, std::vector<Prediction>& predictions)
    {
        VoxelCloud visited;
        visited.Init(vc.n);

        for (int i = 0; i < vc.n; i++)
        for (int j = 0; j < vc.n; j++)
        for (int k = 0; k < vc.n; k++)
        {
            int idx = i + j * vc.n + k * vc.n * vc.n;
            
            // Skip visited/empty voxels
            if (vc.buffer[idx] < 0.5f) continue;
            if (visited.buffer[idx] > 999.f) continue;

            // Set the prediction center (temporarily) at the current voxel
            Prediction prediction;
            prediction.x = (-0.5f + (float)(i)/(float)(vc.n-1)) * extent.width + extent.x0;
            prediction.y = (-0.5f + (float)(j)/(float)(vc.n-1)) * extent.height + extent.y0;
            prediction.theta = (-0.5f + (float)(k)/(float)(vc.n-1)) * extent.depth + extent.theta0;
            prediction.cnt = 1;

            // Mark voxel as visited
            visited.buffer[idx] = 1000.f;

            _PredictDFS(vc, visited, extent, i, j, k, prediction, 0);
            prediction.x /= (float)prediction.cnt;
            prediction.y /= (float)prediction.cnt;
            prediction.theta /= (float)prediction.cnt;

            predictions.push_back(prediction);
        }
    }

    inline void RunVBFDML(std::vector<Segment> scene, BoxExtent& extent, int n, std::vector<Measurement>& measurements,std::vector<Prediction>& predictions, bool bUseGPU)
    {
        VoxelCloud preprocess, mask, tmp;
        preprocess.Init(n);
        mask.Init(n);
        tmp.Init(n);

        PreprocessScene(preprocess, mask, extent, scene, bUseGPU);
        for (auto measurement : measurements)
        {
            MarchingVoxels(tmp, extent, preprocess, measurement.dx, measurement.dy, measurement.dtheta, measurement.dist, bUseGPU);
            Conv3D(tmp, tmp, bUseGPU);
            DoIntersect(mask, mask, tmp, bUseGPU);
        }

        Predict(mask, extent, predictions);
    }

    // Get all possible combinations of k elements from {1,...,n}
    // Based on: https://stackoverflow.com/questions/12991758/creating-all-possible-k-combinations-of-n-items-in-c
    std::vector<std::vector<int>> GetCombinations(int n, int k) {
        std::vector<std::vector<int>> combinations;
        std::string bitmask(k, 1); // K leading 1's
        bitmask.resize(n, 0); // N-K trailing 0's

        do {
            std::vector<int> combination;
            for (int i = 0; i < n; ++i) {
                if (bitmask[i]) combination.push_back(i);
            }
            combinations.push_back(combination);
        } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

        return combinations;
    }

    std::vector<Measurement> _GenerateMeasurements(Arrangement_2& arr, Prediction pred, int k, int dtheta) {
        std::vector<Measurement> measurements;
        float deltaTheta = 0.f;
        for (int i = 0; i < k; i++) {
            COMPUTE_DISTANCE(arr, pred.x, pred.y, pred.theta + deltaTheta);
            Measurement m(0, 0, deltaTheta, dist);
            measurements.push_back(m);
            deltaTheta += dtheta;
        }
        return measurements;
    }

    void RunImprovedVBFDML(std::vector<Prediction>& output, std::vector<Segment> scene, BoxExtent& extent, std::vector<Measurement> measurements, int n, int l, float epsilon) {
        OMP_INIT();
        
        VoxelCloud preprocess, mask;
        std::vector<VoxelCloud> ms;

        preprocess.Init(n); mask.Init(n);
        PreprocessScene(preprocess, mask, extent, scene);

        int i = 0;
        for (auto measurement : measurements) {
            VoxelCloud m; m.Init(n);
            VoxelCloud m_; m_.Init(n);
            MarchingVoxels(m, extent, preprocess, measurement.dx, measurement.dy, measurement.dtheta, measurement.dist);
            Conv3D(m_, m);
            // export the voxel cloud
            // m_.ExportOBJ(fmt::format("m_{}.obj", ++i).c_str(), extent);
            ms.push_back(m_);
        }
        
        // // divide to all permutations of measurements of size k choose l
        auto combinations = GetCombinations(measurements.size(), l);
        std::vector<std::vector<Prediction>> predictions;

        GENERATE_ARRANGEMENT(scene);

        
        // #pragma omp parallel for
        VoxelCloud tmp; tmp.Init(n);
        VoxelCloud m; m.Init(n);
        for (auto combination : combinations) {
            bool first = true;
            for (int idx : combination) {
                if (first) m = ms[idx];
                else {
                    DoIntersect(tmp, m, ms[idx]);
                    m = tmp;
                } 
                first = false;
            }

            // m.ExportOBJ(fmt::format("isect_{}.obj", ++i).c_str(), extent);

            std::vector<Prediction> pred;
            Predict(m, extent, pred);
            predictions.push_back(pred);
        } 
        
        // int ccccnt = 0;
        // for (auto pppp : predictions) {
        //     ccccnt += pppp.size();
        // }
        // fmt::print("Combination: {}\nPredictions: {}\n", combinations.size(), ccccnt);

        // fmt::print("Combinations: {}\n", combinations.size());
        // #pragma omp parallel for collapse(2)
        for (int i = 0; i < combinations.size(); i++) {
            std::vector<Prediction>& pred = predictions[i];
            std::vector<int>& combination = combinations[i];
            if (pred.size() == 0) continue;

            #pragma omp parallel for
            for (auto p : pred) {
                std::vector<Measurement> pred_measurements = _GenerateMeasurements(arr, p, measurements.size(), 2*M_PI / measurements.size());
                int cnt = 0;
                for (int idx : combination) {
                    cnt += abs(pred_measurements[idx].dist - measurements[idx].dist) < epsilon;
                }
                if (cnt >= l-2 || 0) {
                    #pragma omp critical
                    {
                        output.push_back(p);
                    }
                }
            }
        }
    }



}
