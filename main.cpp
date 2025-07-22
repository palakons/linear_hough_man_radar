#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>

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

void findInliers(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                 double inlier_max_dist = 0.5, int min_inliers = 10)
{
    if (cloud->points.size() < 2)
        throw std::runtime_error("Need at least 2 points");

    auto start_time = std::chrono::high_resolution_clock::now();

    pcl::PointXYZ bestPoint1, bestPoint2;
    int countPairsWithOkayInliers = 0;
    int maxInliers = 0;
    int totalPairsChecked = 0;

    for (int i = 0; i < cloud->points.size(); ++i)
    {
        for (int j = i + 1; j < cloud->points.size(); ++j)
        {
            totalPairsChecked++;
            const pcl::PointXYZ &p0 = cloud->points[i];
            const pcl::PointXYZ &p1 = cloud->points[j];

            int inlierCount = countInliers(cloud, p0, p1, inlier_max_dist);
            if (inlierCount >= min_inliers)
            {
                std::cout << "Line from (" << p0.x << ", " << p0.y << ", " << p0.z << ") to ("
                          << p1.x << ", " << p1.y << ", " << p1.z << ") has " << inlierCount
                          << " inliers" << std::endl;
                countPairsWithOkayInliers++;
                if (inlierCount > maxInliers)
                    maxInliers = inlierCount;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Total time: " << duration.count() << " ms" << std::endl
              << "Total pairs: " << totalPairsChecked << std::endl
              << "Average: " << (double)duration.count() / totalPairsChecked << " ms" << std::endl;

    std::cout << "Found " << countPairsWithOkayInliers
              << " pairs of points with at least " << min_inliers << " inliers with inlier threshold "
              << inlier_max_dist << " m." << std::endl
              << "Maximum inliers: " << maxInliers << std::endl;

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

    std::cout << "Loaded " << cloud->width * cloud->height
              << " data points from " << argv[1] << std::endl;

    try
    {
        findInliers(cloud, 0.5, 75);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}