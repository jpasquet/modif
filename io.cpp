#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}
















cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_label(label);
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}


bool ReadImageToStegToDatum(int **A, int rows, int cols, int label, Datum* datum) {
    CVMatToStegToDatum(A,  rows,  cols, datum);
    datum->set_label(label);
return 1;
}





#endif  // USE_OPENCV

bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');





  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}

using namespace std;

bool ReadFistAstroToDatum(double *A, int w, int h, int label, Datum* datum){
  datum->set_channels(1);
  datum->set_height(h);
  datum->set_width(w);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;

  google::protobuf::RepeatedField<float>* D = datum->mutable_float_data();
  vector<float> X(datum_size, 0.0f);
  for(int i=0;i<w*h;i++){
	X[i] = (float)(A[i]);
       double residual = A[i] - (double)(X[i]);
       if(residual>0.1){
		cout<<"perte de precision : "<<residual<<"  "<<A[i]<<endl;
		exit(1);
	}

  }

   for(int i=0;i<X.size();i++){
	D->Add(X[i]);
	}

	return true;
}





int index(int h, int w,int hh, int ww,  int i){
	return (i * hh + (h)) * ww + (w);
}
char ggg(float x){
	int T=3;
	x=x+T;
	if(x<0) return 0;
	if(x>T*2) return 2*T;
	return x;
}







void CVMatToStegToDatum(int **A, int rows, int cols, Datum* datum) {
  int somminmax=4;
  datum->set_channels(5);
  datum->set_height(rows);
  datum->set_width(cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string X(datum_size, ' ');
	
/*
google::protobuf::RepeatedField<float>* D = datum->mutable_float_data();
vector<float> X;
for(int i=0;i<datum_size;i++){
	X.push_back(0.0f);
}
*/





   double temp1[]={0.0,0.0,0.0,0.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0
                   ,0.0,0.0,-1.0,1.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0};



    double temp2[]={0.0,0.0,0.0,0.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0
                   ,0.0,0.0,-1.0,0.0,0.0
                   ,0.0,0.0,0.0,1.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0};

    double temp3[]={ 0.0,0.0,0.0,0.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0
                   ,0.0,0.0,-1.0,0.0,0.0
                   ,0.0,0.0,1.0,0.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0};





    double temp4[]={0.0,0.0,0.0,0.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0
                   ,0.0,0.0,-1.0,0.0,0.0
                   ,0.0,1.0,0.0,0.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0};




        double temp5[]={0.0,0.0,0.0,0.0,0.0
                      , 0.0,0.0,0.0,0.0,0.0
                       ,0.0,1.0,-1.0,0.0,0.0
                       ,0.0,0.0,0.0,0.0,0.0
                       ,0.0,0.0,0.0,0.0,0.0};



    double temp6[]={0.0,0.0,0.0,0.0,0.0
                   ,0.0,1.0,0.0,0.0,0.0
                   ,0.0,0.0,-1.0,0.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0};





    double temp7[]={0.0,0.0,0.0,0.0,0.0
                   ,0.0,0.0,1.0,0.0,0.0
                   ,0.0,0.0,-1.0,0.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0};






    double temp8[]={0.0,0.0,0.0,0.0,0.0
                   ,0.0,0.0,0.0,1.0,0.0
                   ,0.0,0.0,-1.0,0.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0};








    double temp9[]={0.0,0.0,0.0,0.0,0.0
                  , 0.0,0.0,0.0,0.0,0.0
                  , 0.0,1.0,-2.0,1.0,0.0
                  , 0.0,0.0,0.0,0.0,0.0
                  , 0.0,0.0,0.0,0.0,0.0};





    double temp10[]={0.0,0.0,0.0,0.0,0.0
                  , 0.0,1.0,0.0,0.0,0.0
                  , 0.0,0.0,-2.0,0.0,0.0
                  , 0.0,0.0,0.0,1.0,0.0
                 ,  0.0,0.0,0.0,0.0,0.0};



    double temp11[]={0.0,0.0,0.0,0.0,0.0
                  , 0.0,0.0,1.0,0.0,0.0
                  , 0.0,0.0,-2.0,0.0,0.0
                  , 0.0,0.0,1.0,0.0,0.0
                  , 0.0,0.0,0.0,0.0,0.0};



    double temp12[]={0.0,0.0,0.0,0.0,0.0
                 ,  0.0,0.0,0.0,1.0,0.0
                 ,  0.0,0.0,-2.0,0.0,0.0
                  , 0.0,1.0,0.0,0.0,0.0
                  , 0.0,0.0,0.0,0.0,0.0};



    double temp13[]={0.0,0.0,0.0,0.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0
                  , 0.0,1.0,-3.0,1.0,1.0
                  , 0.0,0.0,0.0,0.0,0.0
                  , 0.0,0.0,0.0,0.0,0.0};



    double temp14[]={0.0,0.0,0.0,0.0,0.0
                    ,0.0,1.0,0.0,0.0,0.0
                  ,  0.0,0.0,-3.0,0.0,0.0
                    ,0.0,0.0,0.0,1.0,0.0
                    ,0.0,0.0,0.0,0.0,1.0};




    double temp15[]={0.0,0.0,0.0,0.0,0.0
                  , 0.0,0.0,1.0,0.0,0.0
                 ,  0.0,0.0,-3.0,0.0,0.0
                  , 0.0,0.0,1.0,0.0,0.0
                 ,  0.0,0.0,1.0,0.0,0.0};



    double temp16[]={0.0,0.0,0.0,0.0,0.0
                 ,  0.0,0.0,0.0,1.0,0.0
                 ,  0.0,0.0,-3.0,0.0,0.0
                 ,  0.0,1.0,0.0,0.0,0.0
                 ,  1.0,0.0,0.0,0.0,0.0};


    double temp17[]={0.0,0.0,0.0,0.0,0.0
                  , 0.0,0.0,0.0,0.0,0.0
                   ,1.0,1.0,-3.0,1.0,0.0
                  , 0.0,0.0,0.0,0.0,0.0
                 ,  0.0,0.0,0.0,0.0,0.0};




    double temp18[]={1.0,0.0,0.0,0.0,0.0
                 ,  0.0,1.0,0.0,0.0,0.0
                 ,  0.0,0.0,-3.0,0.0,0.0
                  , 0.0,0.0,0.0,1.0,0.0
                  , 0.0,0.0,0.0,0.0,0.0};




    double temp19[]={0.0,0.0,1.0,0.0,0.0
                  , 0.0,0.0,1.0,0.0,0.0
                  , 0.0,0.0,-3.0,0.0,0.0
                  , 0.0,0.0,1.0,0.0,0.0
                  , 0.0,0.0,0.0,0.0,0.0};



    double temp20[]={0.0,0.0,0.0,0.0,1.0
                  , 0.0,0.0,0.0,1.0,0.0
                   ,0.0,0.0,-3.0,0.0,0.0
                   ,0.0,1.0,0.0,0.0,0.0
                  , 0.0,0.0,0.0,0.0,0.0};


    double temp21[]={0.0,0.0,0.0,0.0,0.0
                   ,0.0,-1.0,2.0,-1.0,0.0
                   ,0.0,2.0,-4.0,2.0,0.0
                   ,0.0,-1.0,2.0,-1.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0};



    double temp22[]={0.0,0.0,0.0,0.0,0.0
                   ,0.0,-1.0,2.0,-1.0,0.0
                   ,0.0,2.0,-4.0,2.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0};



    double temp23[]={0.0,0.0,0.0,0.0,0.0
                   ,0.0,-1.0,2.0,0.0,0.0
                   ,0.0,2.0,-4.0,0.0,0.0
                   ,0.0,-1.0,2.0,0.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0};




    double temp24[]={0.0,0.0,0.0,0.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0
                   ,0.0,2.0,-4.0,2.0,0.0
                   ,0.0,-1.0,2.0,-1.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0};




    double temp25[]={0.0,0.0,0.0,0.0,0.0
                   ,0.0,0.0,2.0,-1.0,0.0
                   ,0.0,0.0,-4.0,2.0,0.0
                  , 0.0,0.0,2.0,-1.0,0.0
                  , 0.0,0.0,0.0,0.0,0.0};



    double temp26[]={-1.0,2.0,-2.0,2.0,-1.0
                   ,2.0,-6.0,8.0,-6.0,2.0
                   ,-2.0,8.0,-12.0,8.0,-2.0
                   ,2.0,-6.0,8.0,-6.0,2.0
                  ,-1.0,2.0,-2.0,2.0,-1.0};

    double temp27[]={-1.0,2.0,-2.0,2.0,-1.0
                   ,2.0,-6.0,8.0,-6.0,2.0
                   ,-2.0,8.0,-12.0,8.0,-2.0
                   ,0.0,0.0,0.0,0.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0};


    double temp28[]={0.0,0.0,-2.0,2.0,-1.0
                   ,0.0,0.0,8.0,-6.0,2.0
                   ,0.0,0.0,-12.0,8.0,-2.0
                   ,0.0,0.0,8.0,-6.0,2.0
                  ,0.0,0.0,-2.0,2.0,-1.0};


    double temp29[]={0.0,0.0,0.0,0.0,0.0
                   ,0.0,0.0,0.0,0.0,0.0
                   ,-2.0,8.0,-12.0,8.0,-2.0
                   ,2.0,-6.0,8.0,-6.0,2.0
                  ,-1.0,2.0,-2.0,2.0,-1.0};


    double temp30[]={-1.0,2.0,-2.0,0.0,0.0
                   ,2.0,-6.0,8.0,0.0,0.0
                  ,-2.0,8.0,-12.0,0.0,0.0
                   ,2.0,-6.0,8.0,0.0,0.0
                  ,-1.0,2.0,-2.0,0.0,0.0};



//temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8,temp9, temp10
//                        ,temp11, temp12, temp13, temp14, temp15, temp16, temp17, temp18, temp19, temp20,
 //                       temp21, temp22, temp23, temp24, temp25, 
    double *alltemp[] = { temp26, temp27, temp28, temp29, temp30};






int edge3[3][3]={{-1,2,-1},
	         {2,-4,2},
	         {-1,2,-1}};

int edge5[5][5]={
		{-1,2,-2, 2, -1},
		{2,-6,8, -6, 2},
		{-2,8,-12, 8, -2},
		{2,-6,8, -6, 2},
		{-1,2,-2, 2, -1}
		};

int edge7[7][7]={
		{-1,2 ,-4, 6, -4, 2, -1},
		{2 ,-4, 6, -8, 6, -4, 2},
		{-4,6 ,-8, 12, -8, 6,-4},
		{6 ,-8,12, -20, 12, -8, 6},
		{-4,6 ,-8, 12, -8, 6,-4},
		{2 ,-4, 6, -8, 6, -4, 2},
		{-1,2 ,-4, 6, -4, 2, -1}
		};





  for (int y = 0; y < rows; ++y) {// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for (int x = 0; x < cols; ++x) {
			/*int max1x = max(A[x-1][ y], A[x+1][y]);
			int max1y = max(A[x][y-1], A[x][y+1]);
			int min1x = min(A[x-1][y], A[x+1][y]);
			int min1y = min(A[x][y-1], A[x][y+1]);

			int maxTri1 = max(A[x-1][y], A[x][y+1]);
			int maxTri2 = max(A[x-1][y], A[x][y-1]);
			int maxTri3 = max(A[x+1][y], A[x][y-1]);
			int maxTri4 = max(A[x+1][y], A[x][y+1]);

			int minTri1 = min(A[x-1][y], A[x][y+1]);
			int minTri2 = min(A[x-1][y], A[x][y-1]);
			int minTri3 = min(A[x+1][y], A[x][y-1]);
			int minTri4 = min(A[x+1][y], A[x][y+1]);


			int sommeX1 = A[x+1][ y] +A[ x-1][ y];
			int sommeY1 = A[x][ y+1]+A[ x][ y-1];
			int sommeD1 = A[x+1][ y+1]+A[ x-1][ y-1];
			int sommeD2 = A[x-1][ y+1]+A[ x+1][ y-1];

			int maxTriangle1 = max(max1x, A[x][ y-1]);
			int maxTriangle2 = max(max1x, A[x][ y+1]);
			int maxTriangle3 = max(max1y, A[x+1][ y]);
			int maxTriangle4 = max(max1y, A[x-1][ y]);
			int minTriangle1 = min(min1x, A[x][ y-1]);
			int minTriangle2 = min(min1x, A[x][ y+1]);
			int minTriangle3 = min(min1y, A[x+1][ y]);
			int minTriangle4 = min(min1y, A[x-1][ y]);

			int minAutour = min(min1x, min1y);
			int maxAutour = max(max1x, max1y);

			int maxDiag1 = max(A[x+1][ y+1], A[x-1][ y-1]);
			int maxDiag2 = max(A[x+1][ y-1], A[x-1][ y+1]);

			int minDiag1 = min(A[x+1][ y+1], A[x-1][ y-1]);
			int minDiag2 = min(A[x+1][ y-1], A[x-1][ y+1]);

			int LHaut = 2*A[x+1][ y]+2*A[x-1][ y]+2*A[x][ y+1]-A[x+1][ y+1]-A[x-1][y+1]  ;
			int LBas  = 2*A[x+1][ y]+2*A[x-1][ y]+2*A[x][ y-1]-A[x+1][ y-1]-A[x-1][ y-1];
			int LGauche = 2*A[x][ y+1] + 2*A[x][ y-1] + 2*A[x+1][ y] - A[x+1][ y+1] - A[x+1][ y-1];
			int LDroit = 2*A[x][ y+1] + 2*A[x][ y-1] + 2*A[x-1][ y] - A[x-1][ y+1] - A[x-1][ y-1];
			*/
			int toto = int(A[x][ y]);


			int ff=0;

			if(somminmax==4){

		for(int nn=0;nn<5;nn++){
			double e5 = 0.0;
			double *temp = alltemp[nn];
			for(int xx=0;xx<5;xx++){
					for(int yy=0;yy<5;yy++){
						int xt = max(0, min(rows-1, x+xx-2));
						int yt = max(0, min(rows-1, y+yy-2));
						e5 += A[xt][ yt] * temp[xx+yy*5];
					}
				}
			X[index( y,x,  datum_height, datum_width,ff++)] = ggg(e5);
		}

			}else if(somminmax==3){
			/*float e3 = 0.0f;
			float e5 = 0.0f;
			float e7 = 0.0f;
			for(int xx=0;xx<3;xx++){
				for(int yy=0;yy<3;yy++){
					int xt = max(0, min(rows-1, x+xx-1));
					int yt = max(0, min(rows-1, y+yy-1));
					e3 += A[xt][ yt] * edge3[xx][yy];
				}
			}	

			for(int xx=0;xx<5;xx++){
				for(int yy=0;yy<5;yy++){
					int xt = max(0, min(rows-1, x+xx-2));
					int yt = max(0, min(rows-1, y+yy-2));
					e5 += A[xt][ yt] * edge5[xx][yy];
				}
			}

			for(int xx=0;xx<7;xx++){
				for(int yy=0;yy<7;yy++){
					int xt = max(0, min(rows-1, x+xx-3));
					int yt = max(0, min(rows-1, y+yy-3));
					e7 += A[xt][ yt] * edge7[xx][yy];
				}
			}


			X[index( y,x,  datum_height, datum_width,ff++)] = e3/4.0;
			X[index( y,x,  datum_height, datum_width,ff++)] = e5/12.0;
			X[index( y,x,  datum_height, datum_width,ff++)] = e7/20.0;

			X[index( y,x,  datum_height, datum_width,ff++)] = (4*toto - max(max(max(LBas, LDroit),LGauche), LHaut)  ) /4.0f;
			X[index( y,x,  datum_height, datum_width,ff++)] = (4*toto - min(min(min(LBas, LDroit),LGauche), LHaut)  ) /4.0f;
*/


}
else			if(somminmax==2){

			//#minmax22hv
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - max1x - min1x);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - max1y - min1y);


			//#minmax24
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTri1 - minTri1);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTri2 - minTri2);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTri3 - minTri3);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTri4 - minTri4);



			//#minmax34hv
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTriangle1- minTriangle1);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTriangle2- minTriangle2);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTriangle3- minTriangle3);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTriangle4- minTriangle4);
	
			//#minmax41 1e
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(toto - maxAutour);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(toto - maxAutour);

			//2eme ordre
			//#minmax21 2d
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(4*toto -   max(sommeX1, sommeY1) -   min(sommeX1, sommeY1)       );


			//#minmax41 2c
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(4*toto - max(max(sommeX1, sommeD1), max(sommeY1, sommeD2)) -min(min(sommeX1, sommeD1), min(sommeY1, sommeD2)) );


			//#minmax24hv
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(4*toto - max(sommeX1, sommeD1)    - min(sommeX1, sommeD1)        );
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(4*toto - max(sommeY1, sommeD1)    - min(sommeY1, sommeD1)        );
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(4*toto - max(sommeX1, sommeD2)    - min(sommeX1, sommeD2)        );
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(4*toto - max(sommeY1, sommeD2)    - min(sommeY1, sommeD2)        );
		

			//#minmax32hv MODIF
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - max(max(sommeX1, sommeY1), sommeD1)            );
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - min(min(sommeX1, sommeY1), sommeD1)            );


			//3eme ordre
			//#minmax24 et minmax26hv
			//X[index( y,x,  datum_height, datum_width,ff++)] =ggg( 2*4*toto - max(LHaut, LBas)- min(LHaut, LBas));
			//X[index( y,x,  datum_height, datum_width,ff++)] =ggg( 2*4*toto - max(LHaut, LGauche)- min(LHaut, LGauche));
			//X[index( y,x,  datum_height, datum_width,ff++)] =ggg( 2*4*toto - max(LHaut, LDroit)- min(LHaut, LDroit));
			//X[index( y,x,  datum_height, datum_width,ff++)] =ggg( 2*4*toto - max(LGauche, LDroit)- min(LGauche, LDroit));
			//X[index( y,x,  datum_height, datum_width,ff++)] =ggg( 2*4*toto - max(LBas, LDroit) - min(LBas, LDroit));


			//#minmax41 E3d
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg( 4*toto - min(min(min(LBas, LDroit),LGauche), LHaut));
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg( 4*toto - min(min(min(LBas, LDroit),LGauche), LHaut));
			
			//#spam11
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg( -4*toto - (A[x+1][ y+1]- A[x-1][y+1]- A[x+1][y-1]- A[x-1][ y-1] + 2* A[x+1][ y]+ 2* A[x-1][ y]+ 2* A[x][ y+1]+ 2* A[x][ y-1])/4.0);

			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg( -12*toto - (A[x+1][ y+1]- A[x-1][y+1]- A[x+1][y-1]- A[x-1][ y-1] + 2* A[x+1][ y]+ 2* A[x-1][ y]+ 2* A[x][ y+1]+ 2* A[x][ y-1])/4.0);

































			}
			else if(somminmax==1){

			//#minmax22hv
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - max1x - min1x);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - max1y - min1y);


			//#minmax24
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTri1 - minTri1);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTri2 - minTri2);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTri3 - minTri3);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTri4 - minTri4);



			//#minmax34hv
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTriangle1- minTriangle1);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTriangle2- minTriangle2);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTriangle3- minTriangle3);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTriangle4- minTriangle4);
	
			//#minmax41 1e
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - minAutour- maxAutour);


			//2eme ordre
			//#minmax21 2d
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(4*toto -   max(sommeX1, sommeY1) -   min(sommeX1, sommeY1)       );


			//#minmax41 2c
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(4*toto - max(max(sommeX1, sommeD1), max(sommeY1, sommeD2)) -min(min(sommeX1, sommeD1), min(sommeY1, sommeD2)) );


			//#minmax24hv
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(4*toto - max(sommeX1, sommeD1)    - min(sommeX1, sommeD1)        );
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(4*toto - max(sommeY1, sommeD1)    - min(sommeY1, sommeD1)        );
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(4*toto - max(sommeX1, sommeD2)    - min(sommeX1, sommeD2)        );
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(4*toto - max(sommeY1, sommeD2)    - min(sommeY1, sommeD2)        );
		

			//#minmax32hv MODIF
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(4*toto - max(max(sommeX1, sommeY1), sommeD1) - min(min(sommeX1, sommeY1), sommeD1)            );



			//3eme ordre
			//#minmax24 et minmax26hv
			//X[index( y,x,  datum_height, datum_width,ff++)] =ggg( 2*4*toto - max(LHaut, LBas)- min(LHaut, LBas));
			//X[index( y,x,  datum_height, datum_width,ff++)] =ggg( 2*4*toto - max(LHaut, LGauche)- min(LHaut, LGauche));
			//X[index( y,x,  datum_height, datum_width,ff++)] =ggg( 2*4*toto - max(LHaut, LDroit)- min(LHaut, LDroit));
			//X[index( y,x,  datum_height, datum_width,ff++)] =ggg( 2*4*toto - max(LGauche, LDroit)- min(LGauche, LDroit));
			//X[index( y,x,  datum_height, datum_width,ff++)] =ggg( 2*4*toto - max(LBas, LDroit) - min(LBas, LDroit));


			//#minmax41 E3d
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*4*toto - min(min(min(LBas, LDroit),LGauche), LHaut)- max(max(max(LBas, LDroit),LGauche), LHaut));


			//#spam11
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(-4*toto - A[x+1][ y+1]- A[x-1][y+1]- A[x+1][y-1]- A[x-1][ y-1] + 2* A[x+1][ y]+ 2* A[x-1][ y]+ 2* A[x][ y+1]+ 2* A[x][ y-1]);
			}
			else{
			/*
//#minmax22hv
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - max1x - min1x);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - max1y - min1y);


			//#minmax24
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTri1 - minTri1);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTri2 - minTri2);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTri3 - minTri3);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTri4 - minTri4);



			//#minmax34hv
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTriangle1- minTriangle1);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTriangle2- minTriangle2);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTriangle3- minTriangle3);
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(2*toto - maxTriangle4- minTriangle4);
	
			//#minmax41 1e
			X[index( y,x,  datum_height, datum_width,ff++)] = ggg(maxAutour-128);
			X[index( y,x,  datum_height, datum_width,ff++)] = ggg(minAutour-128);

			//2eme ordre
			//#minmax21 2d
			X[index( y,x,  datum_height, datum_width,ff++)] = ggg(  max(sommeX1, sommeY1)/2.0  -128   );
			X[index( y,x,  datum_height, datum_width,ff++)] = ggg(  min(sommeX1, sommeY1)/2.0  -128   );




			//#minmax41 2c
			X[index( y,x,  datum_height, datum_width,ff++)] = ggg( max(max(sommeX1, sommeD1), max(sommeY1, sommeD2))/2.0  -128);
			X[index( y,x,  datum_height, datum_width,ff++)] = ggg( min(min(sommeX1, sommeD1), min(sommeY1, sommeD2))/2.0 -128 );

			//#minmax24hv
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(4*toto - max(sommeX1, sommeD1)    - min(sommeX1, sommeD1)        );
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(4*toto - max(sommeY1, sommeD1)    - min(sommeY1, sommeD1)        );
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(4*toto - max(sommeX1, sommeD2)    - min(sommeX1, sommeD2)        );
			//X[index( y,x,  datum_height, datum_width,ff++)] = ggg(4*toto - max(sommeY1, sommeD2)    - min(sommeY1, sommeD2)        );
		

			//#minmax32hv MODIF
			X[index( y,x,  datum_height, datum_width,ff++)] = ggg( max(max(sommeX1, sommeY1), sommeD1)/2.0     -128       );
			X[index( y,x,  datum_height, datum_width,ff++)] = ggg( min(min(sommeX1, sommeY1), sommeD1)/2.0     -128       );


			//#minmax41 E3d
			X[index( y,x,  datum_height, datum_width,ff++)] = ggg( min(min(min(LBas, LDroit),LGauche), LHaut)/4.0   -128 );
			X[index( y,x,  datum_height, datum_width,ff++)] = ggg( max(max(max(LBas, LDroit),LGauche), LHaut)/4.0   -128);

			//#spam11
			X[index( y,x,  datum_height, datum_width,ff++)] = ggg((- A[x+1][ y+1]- A[x-1][y+1]- A[x+1][y-1]- A[x-1][ y-1] + 2* A[x+1][ y]+ 2* A[x-1][ y]+ 2* A[x][ y+1]+ 2* A[x][ y-1])/4.0-128);
			*/
			}


    }
  }

/*
for(int i=0;i<X.size();i++){
	D->Add(X[i]);
}*/
  datum->set_data(X);
}



























#endif  // USE_OPENCV
}  // namespace caffe
