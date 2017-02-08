// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>      /* printf */
#include <stdlib.h> 



#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
using namespace std;  
using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;



DEFINE_string(backend, "lmdb",
              "The backend {lmdb, leveldb} for storing the result");


double* getCaracArray(string chemin, int *w, int *h){
		FILE *ptr_myfile;
		ptr_myfile=fopen(chemin.c_str(),"rb");
		if (!ptr_myfile)
		{
			printf("Unable to open file!");
			return NULL;
		}
		fread(w,sizeof(int),1,ptr_myfile);
		fread(h,sizeof(int),1,ptr_myfile);
		double *z=new double[(*w)*(*h)];

		fread(z,sizeof(double),(*w)*(*h),ptr_myfile);
		fclose(ptr_myfile);
		return z;
}





int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    // Print output to stderr (while still logging)
    FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

    gflags::SetUsageMessage("FITS convert");
    gflags::ParseCommandLineFlags(&argc, &argv, true);



//.EXE   destination   liste_fits

    if (argc < 3) {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
        return 1;
    }


    std::ifstream infile(argv[2]);
    std::vector<std::pair<std::string, int> > lines;
    std::string line;
    size_t pos;
    int label;
    while (std::getline(infile, line)) {
        pos = line.find_last_of(' ');
        label = atoi(line.substr(pos + 1).c_str());
        lines.push_back(std::make_pair(line.substr(0, pos), label));
        LOG(INFO) << line;
    }







    // Create new DB
    scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
    db->Open(argv[1], db::NEW);
    scoped_ptr<db::Transaction> txn(db->NewTransaction());

    // Storing to db
    std::string root_folder(argv[1]);
    Datum datum;
    int count = 0;


        LOG(INFO) << "Somme : "<<lines.size();

    int line_idg=0;
    for (int line_id = 0; line_id < lines.size(); ++line_id) {
        bool status;

	int label = (lines[line_id].second);

        LOG(INFO) << "Processed " << line_id << " id."<<(lines[line_id].first);

	std::string cheminProgConvertion  = "/home/pony/Bureau/DIGITS-digits-4.0/caffe/tools/fits_to_array.py";
	std::string cheminDestination  = "/home/pony/Bureau/DIGITS-digits-4.0/caffe/tools/sortieTest.bin";
	std::string cheminDestinationIn  =lines[line_id].first;
     




        LOG(INFO) << "convertion...";


	system(("python "+cheminProgConvertion+" "+cheminDestinationIn+" "+cheminDestination).c_str());
        

        LOG(INFO) << "get array...";
	int w=0, h=0;
	double *vecteur = getCaracArray(cheminDestination, &w, &h);

        LOG(INFO) << "to DATUM..."<<w<<h;
	ReadFistAstroToDatum(vecteur,  w,  h,  label, &datum);


        // sequential
        string key_str = caffe::format_int(line_idg, 8) + "_" + lines[line_id].first;
        line_idg=line_idg+1;
        // Put in db
        string out;
        CHECK(datum.SerializeToString(&out));
        txn->Put(key_str, out);

        if (count++ % (32*4) == 0) {
            // Commit db
            txn->Commit();
            txn.reset(db->NewTransaction());
            LOG(INFO) << "Processed " << count << " files.";
        }







    }



    // write the last batch
    if (count % 1000 != 0) {
        txn->Commit();
        LOG(INFO) << "Processed " << count << " files.";
    }

    return 0;




}
