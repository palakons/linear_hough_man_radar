#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

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

    // Print first 5 points
    for (size_t i = 0; i < std::min<size_t>(10, cloud->points.size()); ++i)
        std::cout << "    " << cloud->points[i].x
                  << " " << cloud->points[i].y
                  << " " << cloud->points[i].z << std::endl;

    return 0;
}