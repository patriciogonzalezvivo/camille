#include <string>
#include <time.h>

#include "vera/io/obj.h"
#include "vera/io/ply.h"
#include "vera/io/stl.h"
#include "vera/ops/fs.h"
#include "vera/ops/image.h"
#include "vera/ops/pixel.h"

int main(int argc, char **argv) {

    std::string filename = std::string(argv[1]);
    std::string ext = vera::getExt(filename);
    vera::Mesh mesh;

    if ( ext == "ply" || ext == "PLY" )
        vera::loadPLY( filename, mesh );

    else if ( ext == "obj" || ext == "OBJ" )
        vera::loadOBJ( filename, mesh );

    else if ( ext == "stl" || ext == "STL" )
        vera::loadSTL( filename, mesh );

    // mesh.smoothNormals(45.0);
    clock_t start, end;
    start = clock();
    vera::Image img = vera::toSdf(mesh, 6);
    end = clock();
    double duration_sec = double(end-start)/CLOCKS_PER_SEC;

    std::cout << "Took " << duration_sec << "secs" << std::endl;

    filename.erase(filename.length() - ext.length());
    filename += "png";

    img.save(filename);

    return 1;
}
