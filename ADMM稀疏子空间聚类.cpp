
#include<opencv2/opencv.hpp>

//#include<math.h>
#include <opencv2/core/core.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<string>
#include<sstream>
#include<cmath>
using namespace cv;
double img[32][32][32] = { {{1}} };
int st = 0;
int maxIter=200;//迭代次数
double lambda = 1, rou = 1, e = 0.0001,s=0.5;//lambda为拉格朗日乘子，e为训练精度，s为步长
double Lapla[32][32] = { {0} };
double xishuMatrix[32][32] = { {0} };
double g[32][1024];
double l1tidu[32][32];//稀疏矩阵的l1梯度
double gt[1024][32] = { {0} };//g【】【】矩阵的转置矩阵
double res[1024][32] = { {0} };
double l1tidu2[32][1024] = { {0} };//Y-YC的l1梯度,res
double Regular[32][32] = { {0} };//正则化后的稀疏矩阵
double XiangSiDuMatrix[32][32];//相似度矩阵
double D[32][32] = { {0} };
double Dminus1[32][32] = { {0} };//对角矩阵的逆
double I[32][32] = { {0} };//N阶单位矩阵

void ImageExMartrix(String str)//像素点存矩阵中
{
	Mat src, dst;
	src = imread(str);
	if (str.empty())
	{
		return ;
	}
	dst.create(src.size(), src.type);
	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col < src.cols; col++)
		{
			int b = src.at<Vec3b>(row, col)[0];
			int g = src.at<Vec3b>(row, col)[1];
			int r = src.at<Vec3b>(row, col)[2];
			dst.at<Vec3b>(row, col)[0] = b * 0.299 + g * 0.587 + r * 0.114;
			dst.at<Vec3b>(row, col)[1] = b * 0.299 + g * 0.587 + r * 0.114;
			dst.at<Vec3b>(row, col)[2] = b * 0.299 + g * 0.587 + r * 0.114;

		}
	}
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			img[st][i][j] = dst.at<Vec3b>(i, j)[0];//将第st张图片的像素值存到矩阵中
		}
	}
}


double l1fan(double a[][32])//求矩阵的l1范数
{
	double x = 0;
	int m, n;
	n = 32;//列数
	m = sizeof(a) / 32;//行数
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			x += abs(a[i][j]);
		}
	}
	return x;
}

void inputFun()//单位像素点数据转化为行向量，并且依次存于矩阵中
{
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			for (int k = 0; k < 32; k++)
			{
				g[i][j * 32 + k] = img[i][j][k];//将每一个图片的的矩阵变成行向量
			}
		}
	}
}

int sign(double x)//sgn函数
{
	if (x > 0)
		return 1;
	else if (x < 0)
		return -1;
	else return 0;

}
void  FunSign()//求矩阵l1范数的次梯度
{
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			l1tidu[i][j] = sign(xishuMatrix[i][j]);
		}
	}
}

double wuqiongfanshu(double a[][32])//求无穷范数
{
	double max = 0,max1=0;
	for (int i = 0; i < 1024; i++)
	{
		max1 = 0;
		for (int j = 0; j < 32; j++)
		{
			max1 += abs(a[i][j]);
		}
		if (max < max1)
		{
			max = max1;
		}
	}
	return max;
}

double l2fan(double r[][32])//实质为l2范数平方
{
	double l2 = 0;
	for (int i = 0; i < 1024; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			l2 += abs(r[i][j])*abs(r[i][j]);
		}
	}
//	l2 = sqrt(l2);
	return l2;
}

void Lpk()//拉格朗日函数
{
	double Lp = 0;
	FunSign();
	grad1Fun();
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			xishuMatrix[i][j] -= s * l1tidu[i][j];//稀疏矩阵梯度下降
		}
	}MutilMatrix(xishuMatrix, gt);//持续更新（Y-YC）
	for (int i = 0; i < 1024; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			
			res[i][j] -= s * (  sqrt(l2fan(res))+res[i][j]+l1tidu2[i][j]);//Y-YC梯度下降
		}
	}
	double gzhengze;//变量代表正则项的梯度
	gzhengze = 2 * sqrt(l2fan(res))*l1fan(res);//

	//Lp = l1fan(xishuMatrix) + lambda * l1fan2(res) + rou * 0.5*l2fan(res) + gzhengze;//待定，需加正则项的导数
}
void argx()//对稀疏矩阵开始迭代,采用梯度下降法
{
	double L1 = 0,L2 = 0;
	L2 = l1fan(xishuMatrix) + lambda * l1fan(res) + rou * 0.5*l2fan(res);//增广拉格朗日公式
	int cnt = 0;
	while (abs(L2 - L1) > e&&cnt<200)
	{
		FunSign();//求函数次梯度
		
		L1 = L2;
		Lpk();
		MutilMatrix(xishuMatrix, gt);//更新res矩阵
		L2= l1fan(xishuMatrix) + lambda * l1fan(res) + rou * 0.5*l2fan(res);//增广拉格朗日公式
		cnt++;
		diag0();
	}
}
void diag0()//将主对角线元素变为0
{
	for (int i = 0; i < 32; i++)
		xishuMatrix[i][i] = 0;
}
void admm()//利用admm求解稀疏矩阵
{
	//argx();
	MutilMatrix(xishuMatrix, gt);
	//wuqiongfanshu(res);
	double L1 = 0, L2 = 0;
	L2 = l1fan(xishuMatrix)+lambda*l1fan(res)+rou*0.5*l2fan(res);//增广拉格朗日公式

	while (maxIter--&&(abs(L2-L1)>e))//Lrou1-Lrou2<e
	{
		argx();//稀疏矩阵在执行此函数后会更新
		lambda = lambda + rou * l1fan(res);//更新拉格朗日乘子
		L1 = L2;
		MutilMatrix(xishuMatrix, gt);//持续更新（Y-YC）
		L2= l1fan(xishuMatrix) + lambda * l1fan(res) + rou * 0.5*l2fan(res);
		diag0();//使Cii=0
	}
}

void MutilMatrix(double xishuMatrix[][32],double Y[][32])//实现Y-YC
{
	for (int i = 0; i < 1024; i++)//矩阵相乘
	{
		for (int j = 0; j < 32; j++)
		{
			for (int k = 0; k < 32; k++)
			{
				res[i][j] += xishuMatrix[k][j] * Y[i][k];
			}
		}
	}
	for (int i = 0; i < 1024; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			res[i][j] -= Y[i][j];
		}
	}
}
void Tzhuan(double a[][1024], double b[][32])//使得矩阵的列向量来表示图片的数据
{
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 1024; j++)
		{
			b[j][i] = a[i][j];
		}
	}
}

int sgn2(double x)
{

	if (x > 0)
		return 1;
	else if (x < 0)
		return -1;
	else return 0;
}
void grad1Fun()
{
	for (int i = 0; i < 1024; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			l1tidu2[i][j] = sgn2(res[i][j]);
		}
	}
}
double l1fan2(double a[][32])
{
	double x = 0;
	for (int i = 0; i < 1024; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			x += abs(a[i][j]);
		}
	}
	return x;
}
//以上为稀疏矩阵（同样是系数矩阵）的优化迭代部分
double WuqiongLie(double a[32])//取系数矩阵每一列向量的无穷范数
{
	double max = 0;
	for (int i = 0; i < 32; i++)
	{
		if (max <abs( a[i]))
			max = abs(a[i]);
	}
	return max;
}

void  XishuMatrixRegularzation()//相似度矩阵的正则化
{
	double a[32] = { 0 }, b[32] = { 0 }, c[32][32] = { {0} };
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			a[j] = xishuMatrix[j][i];
		}
		b[i]=WuqiongLie(a);//将每一列的无穷范数存储于b数组中

	}
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			Regular[i][j] = xishuMatrix[i][j] / b[j];//系数矩阵正则化
		}
		
	}
}

void like()//构建相似度矩阵
{
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			XiangSiDuMatrix[i][j] = abs(Regular[i][j]) + abs(Regular[j][i]);//W=|C|+|C|t
		}
	}
}

void DuiJiao()//求对角矩阵
{
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			D[i][i] += XiangSiDuMatrix[i][j];
		}
	}
	for (int i = 0; i < 32; i++)
	{
		Dminus1[i][i] = 1 / D[i][i];
		I[i][i] = 1;//作为单位矩阵
	}
}



void LaplacianMatrix()//求解正则化后的拉普拉斯矩阵
{
	double M[32][32] = { {0} };
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			for (int k = 0; k < 32; k++)
			{
				M[i][j] = XiangSiDuMatrix[i][k]*Dminus1[k][j];
			}
		}
	}
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			Lapla[i][j] = I[i][j] - M[i][j];//拉普拉斯矩阵的正则化
		}
	}
}


int main()
{
	String str;
	while (st < 32)
	{
		str = "C:" + ((char)('0' + st)) ;
		str += ".jpg";
		ImageExMartrix(str);
		st++;
	}

	for (int i = 0; i < 32; i++)
	{
			xishuMatrix[i][i] =0 ;//初始化稀疏系数矩阵	
	}
	inputFun();
	Tzhuan(g, gt);
}
