
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
int maxIter=200;//��������
double lambda = 1, rou = 1, e = 0.0001,s=0.5;//lambdaΪ�������ճ��ӣ�eΪѵ�����ȣ�sΪ����
double Lapla[32][32] = { {0} };
double xishuMatrix[32][32] = { {0} };
double g[32][1024];
double l1tidu[32][32];//ϡ������l1�ݶ�
double gt[1024][32] = { {0} };//g�������������ת�þ���
double res[1024][32] = { {0} };
double l1tidu2[32][1024] = { {0} };//Y-YC��l1�ݶ�,res
double Regular[32][32] = { {0} };//���򻯺��ϡ�����
double XiangSiDuMatrix[32][32];//���ƶȾ���
double D[32][32] = { {0} };
double Dminus1[32][32] = { {0} };//�ԽǾ������
double I[32][32] = { {0} };//N�׵�λ����

void ImageExMartrix(String str)//���ص�������
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
			img[st][i][j] = dst.at<Vec3b>(i, j)[0];//����st��ͼƬ������ֵ�浽������
		}
	}


}


double l1fan(double a[][32])//������l1����
{
	double x = 0;
	int m, n;
	n = 32;//����
	m = sizeof(a) / 32;//����
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			x += abs(a[i][j]);
		}
	}
	return x;
}

void inputFun()//��λ���ص�����ת��Ϊ���������������δ��ھ�����
{
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			for (int k = 0; k < 32; k++)
			{
				g[i][j * 32 + k] = img[i][j][k];//��ÿһ��ͼƬ�ĵľ�����������
			}
		}
	}
}



int sign(double x)//sgn����
{
	if (x > 0)
		return 1;
	else if (x < 0)
		return -1;
	else return 0;

}
void  FunSign()//�����l1�����Ĵ��ݶ�
{
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			l1tidu[i][j] = sign(xishuMatrix[i][j]);
		}
	}

}


double wuqiongfanshu(double a[][32])//�������
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

double l2fan(double r[][32])//ʵ��Ϊl2����ƽ��
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

void Lpk()//�������պ���
{
	double Lp = 0;
	FunSign();
	grad1Fun();
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			xishuMatrix[i][j] -= s * l1tidu[i][j];//ϡ������ݶ��½�
		}
	}MutilMatrix(xishuMatrix, gt);//�������£�Y-YC��
	for (int i = 0; i < 1024; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			
			res[i][j] -= s * (  sqrt(l2fan(res))+res[i][j]+l1tidu2[i][j]);//Y-YC�ݶ��½�
		}
	}
	double gzhengze;//����������������ݶ�
	gzhengze = 2 * sqrt(l2fan(res))*l1fan(res);//

	//Lp = l1fan(xishuMatrix) + lambda * l1fan2(res) + rou * 0.5*l2fan(res) + gzhengze;//���������������ĵ���
	
}
void argx()//��ϡ�����ʼ����,�����ݶ��½���
{
	double L1 = 0,L2 = 0;
	L2 = l1fan(xishuMatrix) + lambda * l1fan(res) + rou * 0.5*l2fan(res);//�����������չ�ʽ
	int cnt = 0;
	while (abs(L2 - L1) > e&&cnt<200)
	{
		FunSign();//�������ݶ�
		
		L1 = L2;
		Lpk();
		MutilMatrix(xishuMatrix, gt);//����res����
		L2= l1fan(xishuMatrix) + lambda * l1fan(res) + rou * 0.5*l2fan(res);//�����������չ�ʽ
		cnt++;
		diag0();
	}
}
void diag0()//�����Խ���Ԫ�ر�Ϊ0
{
	for (int i = 0; i < 32; i++)
		xishuMatrix[i][i] = 0;
}
void admm()//����admm���ϡ�����
{
	//argx();
	MutilMatrix(xishuMatrix, gt);
	//wuqiongfanshu(res);
	double L1 = 0, L2 = 0;
	L2 = l1fan(xishuMatrix)+lambda*l1fan(res)+rou*0.5*l2fan(res);//�����������չ�ʽ

	while (maxIter--&&(abs(L2-L1)>e))//Lrou1-Lrou2<e
	{
		argx();//ϡ�������ִ�д˺���������
		lambda = lambda + rou * l1fan(res);//�����������ճ���
		L1 = L2;
		MutilMatrix(xishuMatrix, gt);//�������£�Y-YC��
		L2= l1fan(xishuMatrix) + lambda * l1fan(res) + rou * 0.5*l2fan(res);
		diag0();//ʹCii=0
	}
}

void MutilMatrix(double xishuMatrix[][32],double Y[][32])//ʵ��Y-YC
{
	for (int i = 0; i < 1024; i++)//�������
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
void Tzhuan(double a[][1024], double b[][32])//ʹ�þ��������������ʾͼƬ������
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
//����Ϊϡ�����ͬ����ϵ�����󣩵��Ż���������
double WuqiongLie(double a[32])//ȡϵ������ÿһ�������������
{
	double max = 0;
	for (int i = 0; i < 32; i++)
	{
		if (max <abs( a[i]))
			max = abs(a[i]);
	}
	return max;
}

void  XishuMatrixRegularzation()//���ƶȾ��������
{
	double a[32] = { 0 }, b[32] = { 0 }, c[32][32] = { {0} };
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			a[j] = xishuMatrix[j][i];
		}
		b[i]=WuqiongLie(a);//��ÿһ�е�������洢��b������

	}
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			Regular[i][j] = xishuMatrix[i][j] / b[j];//ϵ����������
		}
		
	}



}

void like()//�������ƶȾ���
{
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			XiangSiDuMatrix[i][j] = abs(Regular[i][j]) + abs(Regular[j][i]);//W=|C|+|C|t
		}
	}

}

void DuiJiao()//��ԽǾ���
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
		I[i][i] = 1;//��Ϊ��λ����
	}
}



void LaplacianMatrix()//������򻯺��������˹����
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
			Lapla[i][j] = I[i][j] - M[i][j];//������˹���������
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
			xishuMatrix[i][i] =0 ;//��ʼ��ϡ��ϵ������
		
	}
	inputFun();
	Tzhuan(g, gt);



}