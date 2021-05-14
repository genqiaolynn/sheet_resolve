#include "CrossDetector.h"

#define PI 3.141592653

using namespace std;
using namespace cv;


const int DIRECTION_FLAG[4] = { DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT };

CrossDetector::CrossDetector(int unit_count/*=15*/, double angle/*=0*/, double angle_unit/*=0.5*/, int cross_break/*=4*/, int cross_burr/*=15*/, int grayvalve/*=180*/, int cross_len/*=20*/, int cross_percent/*=80*/) {
	options.m_CrossArg = unit_count;
	options.m_CrossBreak = cross_break;
	options.m_CrossBurr = cross_burr;
	options.m_CrossGray = grayvalve;
	options.m_CrossLen = cross_len;
	options.m_CrossPercent = cross_percent;
	this->total_angle_count = (unit_count * 2 + 1);
	arglist_t.resize(total_angle_count);
	arglist = arglist_t.data();
	arglist[0] = static_cast<int>(angle);
	for (int i = 1; i <= unit_count; i++) {
		arglist[2 * i - 1] = static_cast<int>(angle + i * angle_unit);
		arglist[2 * i] = static_cast<int>(angle - i * angle_unit);
	}
	/*
	for (int i = 0; i < total_angle_count; i++) {
		cout << " " << arglist[i] << " ";
	}
	system("pause");
	// output： 0 0 0 1 -1 1 -1 2 -2 2 -1 3 -3 ....  7 -7 7 -7
	*/

	buffer_t.resize(2 * 4 * total_angle_count);
	this->buffer = buffer_t.data();
	dx[0] = buffer + total_angle_count * 0;
	dx[1] = buffer + total_angle_count * 1;
	dx[2] = buffer + total_angle_count * 2;
	dx[3] = buffer + total_angle_count * 3;
	dy[0] = buffer + total_angle_count * 4;
	dy[1] = buffer + total_angle_count * 5;
	dy[2] = buffer + total_angle_count * 6;
	dy[3] = buffer + total_angle_count * 7;
	
	//for (int i = 0; i < 4; i++)
	//	cout << "dx:" << dx[i] << " dy:" << dy[i] << endl;
	//system("pause");

	for (int i = 0; i < total_angle_count; i++)
	{
		dx[0][i] = -sin(arglist[i] * PI / 180.);
		dy[0][i] = -cos(arglist[i] * PI / 180.);
		dx[1][i] = sin(arglist[i] * PI / 180.);
		dy[1][i] = cos(arglist[i] * PI / 180.);
		dx[2][i] = -cos(arglist[i] * PI / 180.);
		dy[2][i] = sin(arglist[i] * PI / 180.);
		dx[3][i] = cos(arglist[i] * PI / 180.);
		dy[3][i] = -sin(arglist[i] * PI / 180.);
	}
	/*
	for (int i = 0; i < total_angle_count; i++) {
		for (int y = 0; y < 4; y++) {
			cout << "dx[" << y << "][" << i << "]:" << dx[y][i] << " ";
			cout << "dy[" << y << "][" << i << "]:" << dy[y][i] << " ";
		}
		cout << endl;
	}
	*/
	//system("pause");
}

int CrossDetector::detect(const Mat& _imageGray,const Rect & rect, vector<Cross> & vecCrosses) {
	CV_Assert(_imageGray.channels() == 1);
	Mat imageGray = _imageGray(rect);
	Mat imageBin;
	Mat imageHandle = Mat::zeros(imageGray.size(), imageGray.type());
	adaptiveThreshold(imageGray, imageBin, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 10); // 白底黑子
	GaussianBlur(imageBin, imageBin, cv::Size(3, 3), 2);
	threshold(imageBin, imageBin, 255 * (18 - 3) / 18, 255, THRESH_BINARY);
	
	int step = 1;
	//cout << "step :" << step++ << endl;
	/// 扫描图像
	int iRows = imageGray.rows;
	int iCols = imageGray.cols;
	//if (imageGray.isContinuous()) {
	//	iCols *= iRows;
	//	iRows = 1;
	//}
	//cout << "step :" << step++ << endl;
	uchar * pH, *pB; 
	for (int y = 0; y < iRows; y++) {
		pB = const_cast<uchar*>(imageBin.ptr<uchar>(y));
		pH = const_cast<uchar*>(imageHandle.ptr<uchar>(y));
		for (int x = 0; x < iCols; x++) {
			uchar uB = pB[x];
			//int iB = static_cast<int>(pB[x]);
			uchar uH = pH[x];
			if (uH) continue;
			if (uB) continue;
			bool isCrossF = isCrossPossable(imageBin, x, y, uB);
			if (!isCrossF) continue;
			_Cross cross;
			bool isCrossS = isCross(imageBin, x, y, cross, uB);
			if (isCrossS) {
				int sumx = cross.x;
				int sumy = cross.y;
				int sign_count = 1;
				int sign_ = cross.dir;
				for (int yy = max(y - 5, 0), endyy = min(iRows, y + 5 + 1); yy < endyy; yy++)
				{
					for (int xx = max(x - 5, 0), endxx = min(iCols, x + 5 + 1); xx < endxx; xx++)
					{
						if (xx == x && yy == y)continue;
						if (IMAGE_GRAY_PIX_VALUE(imageHandle, xx, yy))continue;
						_Cross sign2;
						bool isSign_2 = isCross(imageBin, xx, yy, sign2, IMAGE_GRAY_PIX_VALUE(imageHandle, xx, yy));
						if (isSign_2) {
							sumx += sign2.x;
							sumy += sign2.y;
							sign_ |= sign2.dir;
							sign_count++;
							STE_IMAGE_GRAY_PIX_VALUE(imageHandle, xx, yy, 1);
						}
					}
				}
				STE_IMAGE_GRAY_PIX_VALUE(imageHandle, x, y, 1);
				cross.x = rect.x + (int)(sumx / (double)sign_count + 0.5);
				cross.y = rect.y + (int)(sumy / (double)sign_count + 0.5);
				cross.dir = (CROSS_DIR_FLAG)sign_;
				//signs.push_back(sign);
				Cross cross_;
				cross_.x = cross.x;
				cross_.y = cross.y;
				cross_.dir = cross.dir;
				cross_.arg = 0;
				vecCrosses.push_back(cross_);
			}
		}
	}
	return 0;
}


bool CrossDetector::isCrossPossable(Mat & imageBin, int x, int y, const uchar & pix) {
	int width = imageBin.cols; 
	int height = imageBin.rows;
	if (pix)
		return false;
	int top = 0, bottom = 0, left = 0, right = 0;
	do { /// up
		/// width=4pix 的一个竖线
		int ys = std::max(0, y - options.m_CrossLen);
		int ye = std::max(0, y - 1);
		int xs = std::max(0, x - 2);
		int xe = std::min(width - 1, x + 2);
		uchar * p;
		for (int yy = ys; yy <= ye; yy++) {
			p = const_cast<uchar*>(imageBin.ptr<uchar>(yy));
			for (int xx = xs; xx <= xe; xx++) {
				if (!p[xx]) {
					top++; break;
				}
			}
		}
	} while (0);
	do{ /// down
		int ys = std::min(height - 1, y + 1);
		int ye = std::min(height - 1, y + options.m_CrossLen);
		int xs = std::max(0, x - 2);
		int xe = std::min(width - 1, x + 2);
		uchar * p;
		for (int yy = ys; yy <= ye; yy++) {
			p = const_cast<uchar*>(imageBin.ptr<uchar>(yy));
			for (int xx = xs; xx <= xe; xx++) {
				if (!p[xx]) {
					bottom++; break;
				}
			}
		}
	} while (0);
	do { /// left
		int ys = std::max(0, y - 2);
		int ye = std::min(height - 1, y + 2);
		int xs = std::max(0, x - options.m_CrossLen);
		int xe = std::max(0, x - 1);
		uchar * p;
		int len = xe - xs + 1;
		int * pflag = new int[len];
		memset(pflag, 0, sizeof(int)*len);
		for (int yy = ys; yy <= ye; yy++) {
			p = const_cast<uchar*>(imageBin.ptr<uchar>(yy));
			for (int xx = xs; xx <= xe; xx++) {
				if (!p[xx])	pflag[xe - xx] |= 1;
				else pflag[xe - xx] |= 0;
			}
		}
		for (int i = 0; i < len; i++) {
			if (pflag[i]) left++;
		}
		delete[] pflag;
	} while (0);
	do { /// 右
		int ys = std::max(0, y - 2);
		int ye = std::min(height - 1, y + 2);
		int xs = std::min(width - 1, x + 1);
		int xe = std::min(width - 1, x + options.m_CrossLen);
		uchar * p;
		int len = xe - xs + 1;
		int * pflag = new int[len];
		memset(pflag, 0, sizeof(int)*len);
		for (int yy = ys; yy <= ye; yy++) {
			p = const_cast<uchar*>(imageBin.ptr<uchar>(yy));
			for (int xx = xs; xx <= xe; xx++) {
				if (!p[xx])	pflag[xe - xx] |= 1;
				else pflag[xe - xx] |= 0;
			}
		}
		for (int i = 0; i < len; i++) {
			if (pflag[i]) right++;
		}
		delete[] pflag;
	} while (0);
	int value = std::max(5, options.m_CrossLen - 4);
	if ((left > value || right > value) && (top > value || bottom > value))
		return true;
	return false;
}

bool CrossDetector::isCross(Mat & imageBin, int x, int y, _Cross & cross, const uchar & pix) {
	int width = imageBin.cols;
	int height = imageBin.rows;
	int haveburr, totalpoints;
	if (pix) return false;
	bool result = false;
	for (int arg0 = 0; arg0 < 2 * options.m_CrossArg + 1; arg0++) {
		int nowburr = 0;
		int nowCross = 0;
		for (int dir = 0; dir < 4; dir++) {
			int breakPoints = 0;
			int len = 0;
			for (len = 1; len <= options.m_CrossLen; len++) {
				int posx = x + (int)(len*dx[dir][arg0] + 0.5);
				int posy = y + (int)(len*dy[dir][arg0] + 0.5);
				//cout << "pt: (" << posx << "," << posy << ") ";
				if (posx < 0 || posx >= width || posy < 0 || posy >= height) break;
				if (IMAGE_GRAY_PIX_VALUE(imageBin, posx, posy)) {
					breakPoints++;
					if (breakPoints > options.m_CrossBreak) {
						len -= breakPoints;
						break;
					}
				}
				else
					breakPoints = 0;
			}
			if (len > options.m_CrossLen)
				nowCross |= DIRECTION_FLAG[dir];
			else if (len > options.m_CrossBurr)
				nowburr = 1;

		}
		if ((nowCross & 3) && (nowCross & 12)) { /// say: 必须有交叉 不能只是直线
			if (options.m_CrossPercent != 0) { /// say: 周围空白检查
				int blankpoint = 0; /// say: 交叉点周围必须有足够多的白点
				int xx, yy, xx0, yy0;

				//*********************************右上区域检查**********************
				totalpoints = options.m_CrossLen / 2;
				//寻找直线右边界

				for (xx = 1; xx <= options.m_CrossLen / 2; xx++) {
					blankpoint = 0;
					for (yy = -1; yy >= -options.m_CrossLen / 2; yy--) {
						int posx = (int)(x + xx * dx[3][arg0] + 0.5);
						int posy = (int)(y + yy * dy[1][arg0] + 0.5);
						//cout << "pt: (" << posx << "," << posy << ") ";
						if (posx < 0 || posx >= width || posy < 0 || posy >= height)break;
						if (IMAGE_GRAY_PIX_VALUE(imageBin, posx, posy))  blankpoint++;
					}
					if (blankpoint > totalpoints * 0.5) break;
				}
				if (xx > options.m_CrossLen / 2) continue;
				xx0 = xx;
				//寻找直线上边界							

				for (yy = -1; yy >= -options.m_CrossLen / 2; yy--) {
					blankpoint = 0;
					for (xx = 1; xx <= options.m_CrossLen / 2; xx++) {
						int posx = (int)(x + xx * dx[3][arg0] + 0.5);
						int posy = (int)(y + yy * dy[1][arg0] + 0.5);
						if (posx < 0 || posx >= width || posy < 0 || posy >= height)break;
						if (IMAGE_GRAY_PIX_VALUE(imageBin, posx, posy))  blankpoint++;
					}
					if (blankpoint > totalpoints * 0.5) break;
				}
				if (yy < -options.m_CrossLen / 2) continue;
				yy0 = yy;

				blankpoint = 0;      //交叉点周围必须有足够多的空白点
				totalpoints = 0;
				for (yy = 0; yy > -options.m_CrossLen / 2; yy--) {
					for (xx = 0; xx < options.m_CrossLen / 2 + yy; xx++) {
						int posx = (int)(x + (xx0 + xx) * dx[3][arg0] + 0.5);
						int posy = (int)(y + (yy0 + yy) * dy[1][arg0] + 0.5);
						if (posx < 0 || posx >= width || posy < 0 || posy >= height)break;
						totalpoints++;
						if (IMAGE_GRAY_PIX_VALUE(imageBin, posx, posy)) blankpoint++;
					}
				}
				if (blankpoint < totalpoints / 100. * options.m_CrossPercent) continue;   //空白点太少

				//*********************************左上区域检查**********************
				totalpoints = options.m_CrossLen / 2;
				//寻找直线左边界						
				for (xx = -1; xx >= -options.m_CrossLen / 2; xx--) {
					blankpoint = 0;
					for (yy = -1; yy >= -options.m_CrossLen / 2; yy--) {
						int posx = (int)(x + xx * dx[3][arg0] + 0.5);
						int posy = (int)(y + yy * dy[1][arg0] + 0.5);
						if (posx < 0 || posx >= width || posy < 0 || posy >= height)break;
						if (IMAGE_GRAY_PIX_VALUE(imageBin, posx, posy))  blankpoint++;
					}
					if (blankpoint > totalpoints * 0.5) break;
				}
				if (xx < -options.m_CrossLen / 2) continue;
				xx0 = xx;
				//寻找直线上边界													
				for (yy = -1; yy >= -options.m_CrossLen / 2; yy--) {
					blankpoint = 0;
					for (xx = -1; xx >= -options.m_CrossLen / 2; xx--) {
						int posx = (int)(x + xx * dx[3][arg0] + 0.5);
						int posy = (int)(y + yy * dy[1][arg0] + 0.5);
						if (posx < 0 || posx >= width || posy < 0 || posy >= height)break;
						if (IMAGE_GRAY_PIX_VALUE(imageBin, posx, posy))  blankpoint++;
					}
					if (blankpoint > totalpoints * 0.5) break;
				}
				if (yy < -options.m_CrossLen / 2) continue;
				yy0 = yy;

				totalpoints = 0;
				blankpoint = 0;
				for (yy = 0; yy > -options.m_CrossLen / 2; yy--) {
					for (xx = 0; xx > -options.m_CrossLen / 2 - yy; xx--) {
						int posx = (int)(x + (xx0 + xx) * dx[3][arg0] + 0.5);
						int posy = (int)(y + (yy0 + yy) * dy[1][arg0] + 0.5);
						if (posx < 0 || posx >= width || posy < 0 || posy >= height)break;
						totalpoints++;
						if (IMAGE_GRAY_PIX_VALUE(imageBin, posx, posy)) blankpoint++;
					}
				}
				if (blankpoint < totalpoints / 100. * options.m_CrossPercent) continue;   //空白点太少
				//*********************************右下区域检查**********************
				totalpoints = options.m_CrossLen / 2;
				//寻找直线右边界						
				for (xx = 1; xx <= options.m_CrossLen / 2; xx++) {
					blankpoint = 0;
					for (yy = 1; yy <= options.m_CrossLen / 2; yy++) {
						int posx = (int)(x + xx * dx[3][arg0] + 0.5);
						int posy = (int)(y + yy * dy[1][arg0] + 0.5);
						if (posx < 0 || posx >= width || posy < 0 || posy >= height)break;
						if (IMAGE_GRAY_PIX_VALUE(imageBin, posx, posy))  blankpoint++;
					}
					if (blankpoint > totalpoints * 0.5) break;
				}
				if (xx > options.m_CrossLen / 2) continue;
				xx0 = xx;
				//寻找直线下边界							
				for (yy = 1; yy <= options.m_CrossLen / 2; yy++) {
					blankpoint = 0;
					for (xx = 1; xx <= options.m_CrossLen / 2; xx++) {
						int posx = (int)(x + xx * dx[3][arg0] + 0.5);
						int posy = (int)(y + yy * dy[1][arg0] + 0.5);
						if (posx < 0 || posx >= width || posy < 0 || posy >= height)break;
						if (IMAGE_GRAY_PIX_VALUE(imageBin, posx, posy))  blankpoint++;
					}
					if (blankpoint > totalpoints * 0.5) break;
				}
				if (yy > options.m_CrossLen / 2) continue;
				yy0 = yy;
				totalpoints = 0;
				blankpoint = 0;
				for (yy = 0; yy < options.m_CrossLen / 2; yy++) {
					for (xx = 0; xx < options.m_CrossLen / 2 - yy; xx++) {
						int posx = (int)(x + (xx0 + xx) * dx[3][arg0] + 0.5);
						int posy = (int)(y + (yy0 + yy) * dy[1][arg0] + 0.5);
						if (posx < 0 || posx >= width || posy < 0 || posy >= height)break;
						totalpoints++;
						if (IMAGE_GRAY_PIX_VALUE(imageBin, posx, posy)) blankpoint++;
					}
				}
				if (blankpoint < totalpoints / 100. * options.m_CrossPercent) continue;   //空白点太少

				//*********************************左下区域检查**********************
				totalpoints = options.m_CrossLen / 2;
				//寻找直线左边界						
				for (xx = -1; xx >= -options.m_CrossLen / 2; xx--) {
					blankpoint = 0;
					for (yy = 1; yy <= options.m_CrossLen / 2; yy++) {
						int posx = (int)(x + xx * dx[3][arg0] + 0.5);
						int posy = (int)(y + yy * dy[1][arg0] + 0.5);
						if (posx < 0 || posx >= width || posy < 0 || posy >= height)break;
						if (IMAGE_GRAY_PIX_VALUE(imageBin, posx, posy))  blankpoint++;
					}
					if (blankpoint > totalpoints * 0.5) break;
				}
				if (xx < -options.m_CrossLen / 2) continue;
				xx0 = xx;
				//寻找直线下边界													
				for (yy = 1; yy <= options.m_CrossLen / 2; yy++) {
					blankpoint = 0;
					for (xx = -1; xx >= -options.m_CrossLen / 2; xx--) {
						int posx = (int)(x + xx * dx[3][arg0] + 0.5);
						int posy = (int)(y + yy * dy[1][arg0] + 0.5);
						if (posx < 0 || posx >= width || posy < 0 || posy >= height)break;
						if (IMAGE_GRAY_PIX_VALUE(imageBin, posx, posy))  blankpoint++;
					}
					if (blankpoint > totalpoints * 0.5) break;
				}
				if (yy > options.m_CrossLen / 2) continue;
				yy0 = yy;
				totalpoints = 0;
				blankpoint = 0;
				for (yy = 0; yy < options.m_CrossLen / 2; yy++) {
					for (xx = 0; xx > -options.m_CrossLen / 2 + yy; xx--) {
						int posx = (int)(x + (xx0 + xx) * dx[3][arg0] + 0.5);
						int posy = (int)(y + (yy0 + yy) * dy[1][arg0] + 0.5);
						if (posx < 0 || posx >= width || posy < 0 || posy >= height)break;
						totalpoints++;
						if (IMAGE_GRAY_PIX_VALUE(imageBin, posx, posy)) blankpoint++;
					}
				}
				if (blankpoint < totalpoints / 100. * options.m_CrossPercent) continue;   //空白点太少			
			}
			if (nowburr) {
				haveburr = 1;
				break;
			}
			totalpoints = 0;
			cross.dir = (CROSS_DIR_FLAG)nowCross;
			cross.x = x;
			cross.y = y;
			result = true;
			for (int dir = 0; dir < 4; dir++) {
				int breakpoints = 0;
				if ((nowCross & DIRECTION_FLAG[dir]) != 0)	{
					for (int len = 1; len <= 100; len++) {  //按最长边计算角度
						int posx = x + (int)(len * dx[dir][arg0] + 0.5);
						int posy = y + (int)(len * dy[dir][arg0] + 0.5);
						if (posx < 0 || posx >= width || posy < 0 || posy >= height)break;
						totalpoints++;
						if (IMAGE_GRAY_PIX_VALUE(imageBin, posx, posy)) {  //潜在断点
							breakpoints++;
							if (breakpoints > options.m_CrossBreak) break;
						}
						else {
							breakpoints = 0;
						}
					}
					totalpoints -= breakpoints;
				}
			}
		}
	}
	return result;
}

void CrossDetector::draw(Mat & imageCor, const Cross & crossDir,Scalar color /* =Scalar(255,0,0) */, int lineThin /*= 2*/) {
	Point pt(crossDir.x, crossDir.y);
	int len = options.m_CrossLen / 2;
	circle(imageCor, pt, len , color, lineThin, 8, 0);
	switch (crossDir.dir)
	{
	case CROSS_TL: {
		line(imageCor, pt, Point(pt.x, pt.y - len), color, lineThin, 8, 0);
		line(imageCor, pt, Point(pt.x - len, pt.y), color, lineThin, 8, 0);
		break;
	}
	case CROSS_TR: {
		line(imageCor, pt, Point(pt.x, pt.y - len), color, lineThin, 8, 0);
		line(imageCor, pt, Point(pt.x + len, pt.y), color, lineThin, 8, 0);
		break;
	}
	case CROSS_BR: {
		line(imageCor, pt, Point(pt.x, pt.y + len), color, lineThin, 8, 0);
		line(imageCor, pt, Point(pt.x + len, pt.y), color, lineThin, 8, 0);
		break;
	}
	case CROSS_BL: {
		line(imageCor, pt, Point(pt.x, pt.y + len), color, lineThin, 8, 0);
		line(imageCor, pt, Point(pt.x - len, pt.y), color, lineThin, 8, 0);
		break;
	}
	case CROSS_TBL: {
		line(imageCor, pt, Point(pt.x, pt.y + len), color, lineThin, 8, 0);
		line(imageCor, pt, Point(pt.x, pt.y - len), color, lineThin, 8, 0);
		line(imageCor, pt, Point(pt.x - len, pt.y), color, lineThin, 8, 0);
		break;
	}
	case CROSS_TBR: {
		line(imageCor, pt, Point(pt.x, pt.y + len), color, lineThin, 8, 0);
		line(imageCor, pt, Point(pt.x, pt.y - len), color, lineThin, 8, 0);
		line(imageCor, pt, Point(pt.x + len, pt.y), color, lineThin, 8, 0);
		break;
	}
	case CROSS_TLR: {
		line(imageCor, pt, Point(pt.x, pt.y - len), color, lineThin, 8, 0);
		line(imageCor, pt, Point(pt.x - len, pt.y), color, lineThin, 8, 0);
		line(imageCor, pt, Point(pt.x + len, pt.y), color, lineThin, 8, 0);
		break;
	}
	case CROSS_BLR: {
		line(imageCor, pt, Point(pt.x, pt.y + len), color, lineThin, 8, 0);
		line(imageCor, pt, Point(pt.x - len, pt.y), color, lineThin, 8, 0);
		line(imageCor, pt, Point(pt.x + len, pt.y), color, lineThin, 8, 0);
		break;
	}
	case CROSS_TBLR: {
		line(imageCor, pt, Point(pt.x, pt.y + len), color, lineThin, 8, 0);
		line(imageCor, pt, Point(pt.x, pt.y - len), color, lineThin, 8, 0);
		line(imageCor, pt, Point(pt.x - len, pt.y), color, lineThin, 8, 0);
		line(imageCor, pt, Point(pt.x + len, pt.y), color, lineThin, 8, 0);
		break;
	}
	default:
		break;
	}

}