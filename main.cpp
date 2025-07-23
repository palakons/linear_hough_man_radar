#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <eigen3/Eigen/Dense>
#include <limits>
#include <omp.h>
#define EIGEN_USE_MKL_ALL
#define EIGEN_DONT_VECTORIZE
#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
#define EIGEN_NO_DEBUG
#define EIGEN_USING_STD_MATH

double pointToLineDistance(const pcl::PointXYZ &point, const pcl::PointXYZ &p0, const pcl::PointXYZ &p1)
{
    // line
    double dx = p1.x - p0.x;
    double dy = p1.y - p0.y;
    double dz = p1.z - p0.z;

    // vector from p0 to the point
    double px = point.x - p0.x;
    double py = point.y - p0.y;
    double pz = point.z - p0.z;

    // cross product magnitude  = ab sin(theta)
    double crossX = py * dz - pz * dy;
    double crossY = pz * dx - px * dz;
    double crossZ = px * dy - py * dx;
    double crossMagnitude = sqrt(crossX * crossX + crossY * crossY + crossZ * crossZ);

    // line p0-p1's length
    double lineLength = sqrt(dx * dx + dy * dy + dz * dz);

    // distance: b sin(theta)
    return (lineLength > 0) ? crossMagnitude / lineLength : 0;
}

int countInliers(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                 const pcl::PointXYZ &p0,
                 const pcl::PointXYZ &p1,
                 double inlier_max_dist)
{
    int inlierCount = 0;
    for (const auto &point : cloud->points)
        if (pointToLineDistance(point, p0, p1) < inlier_max_dist)
            inlierCount++;
    return inlierCount;
}

std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr> getInliersAndCleaned(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                                                                         const pcl::PointXYZ &p0,
                                                                                                         const pcl::PointXYZ &p1,
                                                                                                         double inlier_max_dist)
{
    auto inliers = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    auto cleanedCloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto &point : cloud->points)
    {
        if (pointToLineDistance(point, p0, p1) < inlier_max_dist)
        {
            inliers->points.push_back(point);
        }
        else
        {
            cleanedCloud->points.push_back(point);
        }
    }
    return {inliers, cleanedCloud};
}
std::pair<pcl::PointXYZ, pcl::PointXYZ> fitLineToPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    pcl::PointXYZ centroid;
    for (const auto &point : cloud->points)
    {
        centroid.x += point.x;
        centroid.y += point.y;
        centroid.z += point.z;
    }
    centroid.x /= cloud->points.size();
    centroid.y /= cloud->points.size();
    centroid.z /= cloud->points.size();

    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (const auto &point : cloud->points)
    {
        Eigen::Vector3f diff(point.x - centroid.x, point.y - centroid.y, point.z - centroid.z);
        covariance += diff * diff.transpose();
    }
    covariance /= cloud->points.size();

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3f direction = svd.matrixU().col(0);

    pcl::PointXYZ closestPoint;
    float minDistance = std::numeric_limits<float>::max();
    for (const auto &point : cloud->points)
    {
        float distance = std::abs(direction.dot(Eigen::Vector3f(point.x, point.y, point.z)));
        if (distance < minDistance)
        {
            minDistance = distance;
            closestPoint = point;
        }
    }

    pcl::PointXYZ unitPoint;
    unitPoint.x = closestPoint.x + direction.x();
    unitPoint.y = closestPoint.y + direction.y();
    unitPoint.z = closestPoint.z + direction.z();

    return {closestPoint, unitPoint};
}

void findLines(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
               double inlier_max_dist = 0.5, int min_inliers = 10, int max_pairs = 10)
{
    if (cloud->points.size() < 2)
        throw std::runtime_error("Need at least 2 points");

    for (int k = 0; k < max_pairs; k++)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        int countPairsWithOkayInliers = 0;
        int maxInliers = 0;
        int totalPairsChecked = 0;
        // declare an array to record p0p1, inlierCount in al array for sortign later
        std::vector<std::tuple<pcl::PointXYZ, pcl::PointXYZ, int>> pairsWithInliers;

#pragma omp parallel reduction(+ : totalPairsChecked)
        {
            std::vector<std::tuple<pcl::PointXYZ, pcl::PointXYZ, int>> localPairs;
#pragma omp for nowait schedule(static)
            for (int i = 0; i < cloud->points.size(); ++i)
            {
                for (int j = i + 1; j < cloud->points.size(); ++j)
                {
                    totalPairsChecked++;
                    const pcl::PointXYZ &p0 = cloud->points[i];
                    const pcl::PointXYZ &p1 = cloud->points[j];
                    int inlierCount = countInliers(cloud, p0, p1, inlier_max_dist);
                    localPairs.emplace_back(p0, p1, inlierCount);
                }
            }
#pragma omp critical
            pairsWithInliers.insert(pairsWithInliers.end(), localPairs.begin(), localPairs.end());
        }

        // sort the pairs by inlier count
        std::sort(pairsWithInliers.begin(), pairsWithInliers.end(),
                  [](const auto &a, const auto &b)
                  { return std::get<2>(a) > std::get<2>(b); });

        const auto &[p0, p1, inlierCount] = pairsWithInliers[0];
        std::cout << "Best pair: "
                  << "P0(" << p0.x << ", " << p0.y << ", " << p0.z << "), "
                  << "P1(" << p1.x << ", " << p1.y << ", " << p1.z << "), "
                  << "Inliers: " << inlierCount << std::endl;
        auto [inliers, cleanedCloud] = getInliersAndCleaned(cloud, p0, p1, inlier_max_dist);
        // copy cleanedCloud to cloud
        *cloud = *cleanedCloud;

        // find line to the inliers
        auto [lineP0, lineP1] = fitLineToPointCloud(inliers);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Total time: " << duration.count() << " ms" << "Total pairs: " << totalPairsChecked << "Average: " << (double)duration.count() / totalPairsChecked << " ms" << std::endl;

        std::cout << "Line P0: (" << lineP0.x << ", " << lineP0.y << ", " << lineP0.z << "), "
                  << "Line P1: (" << lineP1.x << ", " << lineP1.y << ", " << lineP1.z << ")" << std::endl;
    }
    return;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <file.pcd>" << std::endl;
        return 1;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud) == -1)
    {
        std::cerr << "Couldn't read file " << argv[1] << std::endl;
        return 1;
    }

    // std::cout << "Loaded " << cloud->width * cloud->height
    //           << " data points from " << argv[1] << std::endl;

    try
    {
        findLines(cloud, 0.5, 75, 3);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}