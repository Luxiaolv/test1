/*
#include <iostream>
#include <string>
#include <onnxruntime_cxx_api.h>
#include<opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;

int main()
{
    string labels[] = { "black-point"};
    cv::dnn::Net net = cv::dnn::readNetFromONNX("E:\\workspace\\work\\yolov5-master\\yolov5-master\\runs\\train\\exp3\\weights\\best.onnx");  // 加载模型
    cv::Mat image = cv::imread("E:\\workspace\\work\\yolov5-master\\datasets\\black-point-data\\images\\val\\008.bmp", 1);  // 读取图片
    cv::Mat blob = cv::dnn::blobFromImage(image, (double)(1.0 / 225.0), cv::Size(), cv::Scalar(225, 225, 225), true, false, CV_32F);  // 由图片加载数据 还可以进行缩放、归一化等预处理操作
    net.setInput(image);  // 设置模型输入
    cv::Mat predict = net.forward(); // 推理结果

    double minValue, maxValue;
    cv::Point minIdx, maxIdx;
    cv::minMaxLoc(predict, &minValue, &maxValue, &minIdx, &maxIdx);

    string res = labels[maxIdx.x];

    return 0;
}
*/

#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>

using namespace cv;
using namespace dnn;
using namespace std;


struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	float objThreshold;  //Object Confidence threshold
	string modelpath;
};

int endsWith(string s, string sub) {
	return s.rfind(sub) == (s.length() - sub.length()) ? 1 : 0;
}

const float anchors_640[3][6] = { { 5.0, 6.0, 8.0, 14.0, 15.0, 11.0 },
				  {30.0,  61.0, 62.0,  45.0,  59.0,  119.0},
				  {116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };

const float anchors_1280[4][6] = { {19, 27, 44, 40, 38, 94},
								   {96, 68, 86, 152, 180, 137},
								   {140, 301, 303, 264, 238, 542},
				   {436, 615, 739, 380, 925, 792} };

class YOLO
{
public:
	YOLO(Net_config config);
	void detect(Mat& frame);
private:
	float* anchors;
	int num_stride;
	int inpWidth;
	int inpHeight;
	vector<string> class_names;
	int num_class;

	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	const bool keep_ratio = true;
	Net net;
	void drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid);
	Mat resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left);
	Mat letterbox(const cv::Mat& im, int* newh, int* neww, int* top, int* left,
		cv::Vec3b color = cv::Vec3b(114, 114, 114), bool auto_pad = true, bool scaleFill = false, bool scaleup = true, int stride = 32);
};

YOLO::YOLO(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;

	this->net = readNet(config.modelpath);
	ifstream ifs("class.names.txt");
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();

	if (endsWith(config.modelpath, "6.onnx"))
	{
		anchors = (float*)anchors_1280;
		this->num_stride = 4;
		this->inpHeight = 1280;
		this->inpWidth = 1280;
	}
	else
	{
		anchors = (float*)anchors_640;
		this->num_stride = 3;
		this->inpHeight = 640;
		this->inpWidth = 640;
	}
}

cv::Mat YOLO::letterbox(const cv::Mat& im, int* newh, int* neww, int* top, int* left,
	cv::Vec3b color, bool auto_pad, bool scaleFill, bool scaleup, int stride) {

	*newh = this->inpHeight;
	*neww = this->inpWidth;
	cv::Size shape = im.size(); // current shape [height, width]  

	// Scale ratio (new / old)  
	double rw = static_cast<double>(this->inpWidth) / shape.width;
	double rh = static_cast<double>(this->inpHeight) / shape.height;
	double r_min = std::min(rw, rh);
	double r = (scaleup) ? std::min(r_min, 1.0) : r_min;

	// Compute padding  
	cv::Size new_unpad(static_cast<int>(round(shape.width * r)), static_cast<int>(round(shape.height * r)));
	int dw = this->inpWidth - new_unpad.width;
	int dh = this->inpHeight - new_unpad.height;
	cv::Size new_shape = new_unpad;
	if (auto_pad) { // minimum rectangle  
		dw = dw % stride;
		dh = dh % stride;
	}
	else if (scaleFill) { // stretch  
		dw = 0;
		dh = 0;
		new_unpad = new_shape;
		r = std::max(rw, rh); // recompute ratio  
	}

	dw /= 2;
	dh /= 2;

	cv::Mat im_resized;
	if (new_unpad != shape) { // resize  
		cv::resize(im, im_resized, new_unpad, 0, 0, cv::INTER_LINEAR);
	}
	else {
		im_resized = im.clone();
	}
	*newh = new_unpad.height;
	*neww = new_unpad.width;
	*top = static_cast<int>(round(dh - 0.1));
	*left = static_cast<int>(round(dw - 0.1));

	cv::Mat im_padded;
	cv::copyMakeBorder(im_resized, im_padded, *top, static_cast<int>(round(dh + 0.1)),
		*left, static_cast<int>(round(dw + 0.1)),
		cv::BORDER_CONSTANT, color);
	

	return im_padded;
}

Mat YOLO::resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void YOLO::drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid)   // Draw the predicted bounding box
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 2);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	label = this->class_names[classid] + ":" + label;

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
}

void YOLO::detect(Mat& frame)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;
	//Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);
	Mat dstimg = this->letterbox(frame, &newh, &neww, &padh, &padw);
	Mat blob = blobFromImage(dstimg, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
	auto start = std::chrono::high_resolution_clock::now();
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
	// 结束时间点
	auto finish = std::chrono::high_resolution_clock::now();
	// 计算耗时，并将其转换为秒
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "modle time: " << elapsed.count() << "s\n";

	int num_proposal = outs[0].size[1];
	int nout = outs[0].size[2];
	if (outs[0].dims > 2)
	{
		outs[0] = outs[0].reshape(0, num_proposal);
	}
	/////generate proposals
	vector<float> confidences;
	vector<Rect> boxes;
	vector<int> classIds;
	float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
	int n = 0, q = 0, i = 0, j = 0, row_ind = 0; ///xmin,ymin,xamx,ymax,box_score,class_score
	float* pdata = (float*)outs[0].data;
	for (n = 0; n < this->num_stride; n++)   ///特征图尺度
	{
		const float stride = pow(2, n + 3);
		int num_grid_x = (int)ceil((this->inpWidth / stride));
		int num_grid_y = (int)ceil((this->inpHeight / stride));
		for (q = 0; q < 3; q++)    ///anchor
		{
			const float anchor_w = this->anchors[n * 6 + q * 2];
			const float anchor_h = this->anchors[n * 6 + q * 2 + 1];
			for (i = 0; i < num_grid_y; i++)
			{
				for (j = 0; j < num_grid_x; j++)
				{
					float box_score = pdata[4];
					if (box_score > this->objThreshold)
					{
						Mat scores = outs[0].row(row_ind).colRange(5, nout);
						Point classIdPoint;
						double max_class_socre;
						// Get the value and location of the maximum score
						minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
						max_class_socre *= box_score;
						if (max_class_socre > this->confThreshold)
						{
							const int class_idx = classIdPoint.x;
							//float cx = (pdata[0] * 2.f - 0.5f + j) * stride;  ///cx
							//float cy = (pdata[1] * 2.f - 0.5f + i) * stride;   ///cy
							//float w = powf(pdata[2] * 2.f, 2.f) * anchor_w;   ///w
							//float h = powf(pdata[3] * 2.f, 2.f) * anchor_h;  ///h

							float cx = pdata[0];  ///cx
							float cy = pdata[1];   ///cy
							float w = pdata[2];   ///w
							float h = pdata[3];  ///h

							int left = int((cx - padw - 0.5 * w) * ratiow);
							int top = int((cy - padh - 0.5 * h) * ratioh);

							confidences.push_back((float)max_class_socre);
							boxes.push_back(Rect(left, top, (int)(w * ratiow), (int)(h * ratioh)));
							classIds.push_back(class_idx);
						}
					}
					row_ind++;
					pdata += nout;
				}
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		this->drawPred(confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame, classIds[idx]);
	}


	// NMS
	/*std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
	for (size_t i = 0; i < indexes.size(); i++) {
		int index = indexes[i];
		int idx = classIds[index];
		cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
		cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
			cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
		cv::putText(frame, class_names[idx], cv::Point(boxes[index].tl().x, boxes[index].tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));
	}*/


}

int main()
{
	Net_config yolo_nets = { 0.7, 0.45, 0.25, "best.onnx" };
	YOLO yolo_model(yolo_nets);
	//YOLO yolo_model = YOLO(yolo_nets);
	string imgpath = "008.bmp";
	Mat srcimg = imread(imgpath);
	auto start = std::chrono::high_resolution_clock::now();
	yolo_model.detect(srcimg);
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Elapsed time: " << elapsed.count() << "s\n";
	imwrite("./res.bmp", srcimg);

	/*int n = 588;
	for (int i = 1; i <= n; i++) {
		string s = to_string(i) + ".png";
		string imgpath = "F://test//p1//yanfa2//bh//cc//" + s;
		cout << imgpath << endl;

		Mat srcimg = imread(imgpath);
		yolo_model.detect(srcimg);
		imwrite("F://test//p2//yanfa2//bh//cc//" + s, srcimg);
	}*/

	
}
