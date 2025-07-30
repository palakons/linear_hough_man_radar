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
#include <fstream>
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
    centroid.x = 0;
    centroid.y = 0;
    centroid.z = 0;
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

    pcl::PointXYZ endpoint;
    endpoint.x = centroid.x + direction.x();
    endpoint.y = centroid.y + direction.y();
    endpoint.z = centroid.z + direction.z();

    return {centroid, endpoint};
}

struct LineDetectionResult
{
    pcl::PointXYZ p0;
    pcl::PointXYZ p1;
    int inliers;
    pcl::PointCloud<pcl::PointXYZ>::Ptr inlierCloud;
    pcl::PointXYZ lineP0;
    pcl::PointXYZ lineP1;
    float lapseTime;
};

std::vector<LineDetectionResult> findLines(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                           double inlier_max_dist = 0.5, int max_pairs = 10)
{
    std::vector<LineDetectionResult> results;
    if (cloud->points.size() < 2)
        throw std::runtime_error("Need at least 2 points");

    pcl::PointCloud<pcl::PointXYZ>::Ptr workingCloud(new pcl::PointCloud<pcl::PointXYZ>(*cloud));

    for (int k = 0; k < max_pairs; k++)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        int totalPairsChecked = 0;
        std::vector<std::tuple<pcl::PointXYZ, pcl::PointXYZ, int>> pairsWithInliers;

#pragma omp parallel reduction(+ : totalPairsChecked)
        {
            std::vector<std::tuple<pcl::PointXYZ, pcl::PointXYZ, int>> localPairs;
#pragma omp for nowait
            for (int i = 0; i < workingCloud->points.size(); ++i)
            {
                for (int j = i + 1; j < workingCloud->points.size(); ++j)
                {
                    totalPairsChecked++;
                    const pcl::PointXYZ &p0 = workingCloud->points[i];
                    const pcl::PointXYZ &p1 = workingCloud->points[j];
                    int inlierCount = countInliers(workingCloud, p0, p1, inlier_max_dist);
                    localPairs.emplace_back(p0, p1, inlierCount);
                }
            }
#pragma omp critical
            pairsWithInliers.insert(pairsWithInliers.end(), localPairs.begin(), localPairs.end());
        }
        std::cout << "Total pairs checked: " << totalPairsChecked << " length of pairsWithInliers: " << pairsWithInliers.size() << std::endl;
        if (pairsWithInliers.empty())
        {
            std::cout << "No pairs with inliers found. Stopping." << std::endl;
            break;
        }
        auto bestPairIt = std::max_element(
            pairsWithInliers.begin(), pairsWithInliers.end(),
            [](const auto &a, const auto &b)
            {
                return std::get<2>(a) < std::get<2>(b);
            });
        const auto &[p0, p1, inlierCount] = *bestPairIt;
        std::cout << "Best pair: (" << p0.x << ", " << p0.y << ", " << p0.z << ") and ("
                  << p1.x << ", " << p1.y << ", " << p1.z << ") with inliers: " << inlierCount << std::endl;
        auto [inliersCloud, cleanedCloud] = getInliersAndCleaned(workingCloud, p0, p1, inlier_max_dist);
        // puch in only if inlierCount > 1
        if (inlierCount < 2)
        {
            std::cout << "Not enough inliers found for pair (" << p0.x << ", " << p0.y << ", " << p0.z << ") and ("
                      << p1.x << ", " << p1.y << ", " << p1.z << "). Skipping." << std::endl;
            continue;
        }
        auto [lineP0, lineP1] = fitLineToPointCloud(inliersCloud);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = end_time - start_time;
        // Store result
        results.push_back(LineDetectionResult{p0, p1, inlierCount, inliersCloud, lineP0, lineP1, elapsed.count()});

        // Update workingCloud for next iteration
        *workingCloud = *cleanedCloud;
    }
    return results;
}

void writeCloudAndResultsToJson(const std::string &filename,
                                const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                const std::vector<LineDetectionResult> &results)
{
    std::ofstream out(filename);

    out << "{\n";
    // Write points
    out << "  \"points\": [\n";
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        const auto &pt = cloud->points[i];
        out << "    {\"x\": " << pt.x << ", \"y\": " << pt.y << ", \"z\": " << pt.z << "}";
        if (i + 1 < cloud->points.size())
            out << ",";
        out << "\n";
    }
    out << "  ],\n";
    // Write results
    out << "  \"lines\": [\n";
    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto &res = results[i];
        out << "    {\n";
        out << "      \"p0\": {\"x\": " << res.p0.x << ", \"y\": " << res.p0.y << ", \"z\": " << res.p0.z << "},\n";
        out << "      \"p1\": {\"x\": " << res.p1.x << ", \"y\": " << res.p1.y << ", \"z\": " << res.p1.z << "},\n";
        out << "      \"inliers\": " << res.inliers << ",\n";
        out << "      \"lapseTime\": " << res.lapseTime << ",\n";
        out << "      \"lineP0\": {\"x\": " << res.lineP0.x << ", \"y\": " << res.lineP0.y << ", \"z\": " << res.lineP0.z << "},\n";
        out << "      \"lineP1\": {\"x\": " << res.lineP1.x << ", \"y\": " << res.lineP1.y << ", \"z\": " << res.lineP1.z << "},\n";
        // Write inlier points for this line
        out << "      \"inlierCloud\": [\n";
        for (size_t j = 0; j < res.inlierCloud->points.size(); ++j)
        {
            const auto &ipt = res.inlierCloud->points[j];
            out << "        {\"x\": " << ipt.x << ", \"y\": " << ipt.y << ", \"z\": " << ipt.z << "}";
            if (j + 1 < res.inlierCloud->points.size())
                out << ",";
            out << "\n";
        }
        out << "      ]\n";
        out << "    }";
        if (i + 1 < results.size())
            out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    out.close();
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

    // keep only first 10 points
    // #declare new cloud

    if (cloud->points.size() > 10 && 0)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr reducedCloud(new pcl::PointCloud<pcl::PointXYZ>);
        std::sample(cloud->points.begin(), cloud->points.end(),
                    std::back_inserter(reducedCloud->points),
                    10, std::mt19937{std::random_device{}()});
        cloud = reducedCloud;
    }

    // Generate output JSON filename based on input filename
    std::string inputFile(argv[1]);
    std::string outputFile = inputFile;
    // Remove directory path
    size_t lastSlash = outputFile.find_last_of("/\\");
    if (lastSlash != std::string::npos)
        outputFile = outputFile.substr(lastSlash + 1);
    // Remove .pcd extension if present
    size_t ext = outputFile.rfind(".pcd");
    if (ext != std::string::npos)
        outputFile = outputFile.substr(0, ext);
    outputFile = "../public/json/"+outputFile+".json";

    //if the output file already exists, return 
    std::ifstream checkFile(outputFile);
    if (checkFile.good())
    {
        std::cerr << "Output file already exists: " << outputFile << ". Please remove it before running again." << std::endl;
        return 1;
    }

    try
    {
        auto results = findLines(cloud, 0.5, 30);
        writeCloudAndResultsToJson(outputFile, cloud, results);
        std::cout << "Wrote results to: " << outputFile << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

