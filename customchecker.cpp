// Start of HEAD
#include <map>
#include <cmath>
#include <cstdio>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <jsoncpp/json/json.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;
using namespace Json;
//Description : https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
//Higher Value of PSNR Denotes more Matching
	double
getPSNR (const Mat & I1, const Mat & I2)
{
	Mat s1;
	absdiff (I1, I2, s1);		// |I1 - I2|
	s1.convertTo (s1, CV_32F);	// cannot make a square on 8 bits
	s1 = s1.mul (s1);		// |I1 - I2|^2
	Scalar s = sum (s1);		// sum elements per channel
	double sse = s.val[0] + s.val[1] + s.val[2];	// sum channels
	if (sse <= 1e-10)		// for small values return zero
		return 0;
	else
	{
		double mse = sse / (double) (I1.channels () * I1.total ());
		double psnr = 10.0 * log10 ((255 * 255) / mse);
		return psnr;
	}
}
//Description : http://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf
//Higher Value of Similarity Denotes mroe matching
	Scalar
getMSSIM (const Mat & i1, const Mat & i2)
{
	const double C1 = 6.5025, C2 = 58.5225;
	/***************************** INITS **********************************/
	int d = CV_32F;
	Mat I1, I2;
	i1.convertTo (I1, d);		// cannot calculate on one byte large values
	i2.convertTo (I2, d);
	Mat I2_2 = I2.mul (I2);	// I2^2
	Mat I1_2 = I1.mul (I1);	// I1^2
	Mat I1_I2 = I1.mul (I2);	// I1 * I2
	/***********************PRELIMINARY COMPUTING ******************************/
	Mat mu1, mu2;			//
	GaussianBlur (I1, mu1, Size (11, 11), 1.5);
	GaussianBlur (I2, mu2, Size (11, 11), 1.5);
	Mat mu1_2 = mu1.mul (mu1);
	Mat mu2_2 = mu2.mul (mu2);
	Mat mu1_mu2 = mu1.mul (mu2);
	Mat sigma1_2, sigma2_2, sigma12;
	GaussianBlur (I1_2, sigma1_2, Size (11, 11), 1.5);
	sigma1_2 -= mu1_2;
	GaussianBlur (I2_2, sigma2_2, Size (11, 11), 1.5);
	sigma2_2 -= mu2_2;
	GaussianBlur (I1_I2, sigma12, Size (11, 11), 1.5);
	sigma12 -= mu1_mu2;
	///////////////////////////////// FORMULA ////////////////////////////////
	Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul (t2);		// t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul (t2);		// t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
	Mat ssim_map;
	divide (t3, t1, ssim_map);	// ssim_map =  t3./t1;
	Scalar mssim = mean (ssim_map);	// mssim = average of ssim map
	return mssim;
}
// Compare two images by getting the L2 error (square-root of sum of squared error).
	double
getSimilarity (const Mat A, const Mat B)
{
	if (A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols)
	{
		// Calculate the L2 relative error between the 2 images.
		double errorL2 = norm (A, B, CV_L2);
		// Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
		double similarity = errorL2 / (double) (A.rows * A.cols);
		return similarity;
	}
	else
	{
		//cout << "WARNING: Images have a different size in 'getSimilarity()'." << endl;
		return 100000000.0;	// Return a bad value
	}
}
//Assumes region of interest is marked as White(R=255,G=255,B=255) and others as Black(R=0,B=0,G=0)
//Can be used for edge detection/foreground background segmentation/ object detection type of examples
	Scalar
pixelSimilarity (Mat img1, Mat img2)
{
	int fp = 0, fn = 0, tp = 0, tn = 0;
	double precision, recall, fmeasure;
	for (int i = 0; i < img1.rows; i++)
	{
		for (int j = 0; j < img1.cols; j++)
		{
			int bimg1, gimg1, rimg1, bimg2, rimg2, gimg2;
			bimg1 = img1.at < cv::Vec3b > (i, j)[0];
			gimg1 = img1.at < cv::Vec3b > (i, j)[1];
			rimg1 = img1.at < cv::Vec3b > (i, j)[2];
			bimg2 = img2.at < cv::Vec3b > (i, j)[0];
			gimg2 = img2.at < cv::Vec3b > (i, j)[1];
			rimg2 = img2.at < cv::Vec3b > (i, j)[2];
			if (bimg1 == 255 && gimg1 == 255 & rimg1 == 255)
			{
				if (bimg2 == 255 && gimg2 == 255 && rimg2 == 255)
					tp++;
				else
					fp++;
			}
			if (bimg2 == 255 && gimg2 == 255 & rimg2 == 255)
			{
				if (bimg1 == 255 && gimg1 == 255 && rimg1 == 255)
					tn++;
				else
					fn++;
			}
		}
	}
	//cout << tp << " " << fp << " " << tn << " " << fn << endl;
	precision = 1.0 * tp / (tp + fp);
	recall = 1.0 * tp / (tp + fn);
	fmeasure = 2.0 * tp / (2 * tp + fp + fn);
	//cout<<"Precision: "<<precision<<" Recall: "<<recall<<" F Measure "<<fmeasure<<endl;
	Scalar output;
	output.val[0] = precision;
	output.val[1] = recall;
	output.val[2] = fmeasure;
	return output;
}
Scalar getHistScore(Mat img1,Mat img2)
{
Scalar score;
 /// Convert to HSV
Mat hsv_1;
Mat hsv_2;
    cvtColor( img1, hsv_1, COLOR_BGR2HSV );
    cvtColor( img2, hsv_2, COLOR_BGR2HSV );
  

    
    int h_bins = 50; int s_bins = 60;
    int histSize[] = { h_bins, s_bins };

    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };

    const float* ranges[] = { h_ranges, s_ranges };

    // Use the o-th and 1-st channels
    int channels[] = { 0, 1 };
MatND hist_1,hist_2;
 calcHist( &hsv_1, 1, channels, Mat(), hist_1, 2, histSize, ranges, true, false );
    normalize( hist_1, hist_1, 0, 1, NORM_MINMAX, -1, Mat() );
 calcHist( &hsv_2, 1, channels, Mat(), hist_2, 2, histSize, ranges, true, false );
    normalize( hist_2, hist_2, 0, 1, NORM_MINMAX, -1, Mat() );
  for( int i = 0; i < 4; i++ )
    {
        int compare_method = i;
         score.val[i] =compareHist( hist_1, hist_2, compare_method );
       
cout<<"Method= "<<i<<" Score: "<<score<<endl;
    }
return score;
}

	int
main (int argc, char **argv)
{
	// Input parameters
	if (argc < 2)
		return -1;
	cout << argv[1] << endl;
	Mat img1 = imread (argv[1], CV_LOAD_IMAGE_COLOR);
	Mat imgL;
	for (int i = 1; i < 101; i = i + 2)
		blur (img1, imgL, Size (i, i));
	imshow ("opencvtest1", img1);
	waitKey (10);
	imshow ("opencvtest2", imgL);
	waitKey (10);
	Scalar mssimV = getMSSIM (img1, imgL);
	cout << "MSSIM: "<< " Red " << setiosflags (ios::fixed) << setprecision (2) << mssimV.val[2] *100 << "%" << " Green " << setiosflags(ios::fixed) << setprecision (2) <<mssimV.val[1] *100 << "%" << " Blue "<< setiosflags (ios::fixed) << setprecision (2) <<	mssimV.val[0] * 100 << "%"<<endl;
	
	cout << "PSNR:" << getPSNR (img1, imgL) << endl;
	Scalar result = pixelSimilarity (img1, imgL);
	cout << "Pixel Similarity: " << result[0] << " " << result[1] << " " <<	result[2] << endl;
	cout << "L2 Similarity: " << getSimilarity (img1, imgL) << endl;
Scalar Hist_score=getHistScore(img1,imgL);
cout<<Hist_score.rows<<endl;
cout<<"Histogram Matching Score: ";
for(int i=0;i<Hist_score.rows;i++)
cout<<Hist_score[i]<<" ";
cout<<endl;
	return 0;
}
