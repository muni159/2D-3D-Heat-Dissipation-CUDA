#include<iostream>
#include<fstream>
#include<string>
#include<algorithm>
#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<iomanip>
#include <limits>
#define BLOCK_SIZE 8
using namespace std;
__global__ void kernel(float *Told,float *Tnew,int *dim,float *k_val,float *sp,int *grid_w,int *grid_h,int *grid_d,int *pn) 
{
 if (*dim==2)
	{
		__shared__ float Mat[BLOCK_SIZE+2][BLOCK_SIZE+2];
		int j = blockIdx.x * blockDim.x + threadIdx.x;
		int i = blockIdx.y * blockDim.y + threadIdx.y;
		int jmat = threadIdx.x+1;
		int imat = threadIdx.y+1;

		if (i<=*grid_h && j<=*grid_w)
		{
			int lengthj=(blockIdx.x==(int)*grid_w/BLOCK_SIZE)? *grid_w%BLOCK_SIZE :BLOCK_SIZE;
			int lengthi=(blockIdx.y==(int)*grid_h/BLOCK_SIZE)? *grid_h%BLOCK_SIZE :BLOCK_SIZE;
			Mat[imat][jmat]=Told[i*(*grid_h)+j];

			if (threadIdx.x < 1)
			{
				Mat[imat][jmat-1]=(j<1)?  *(Told+i*(*grid_h)+(j)): *(Told+i*(*grid_h)+j-1);  /// check j or j-1
				Mat[imat][jmat + lengthj]=(j >= *grid_w-lengthj)? *(Told+i*(*grid_h)+j+lengthj-1):*(Told+i*(*grid_h)+j+lengthj);	
			}

			 if (threadIdx.y < 1)
			{
				Mat[imat-1][jmat]= (i < 1)? *(Told+(i)*(*grid_h)+j): *(Told+(i-1)*(*grid_h)+j);  /// check j or j-1
				Mat[imat + lengthi][jmat]=(i >= (*grid_h)-lengthi)? *(Told+(i+lengthi-1)*(*grid_h)+j): *(Told+(i+lengthi)*(*grid_h)+j);
			}
			__syncthreads();
			
			if (i<*grid_h && j<*grid_w)
			{
				*(Tnew+i*(*grid_h)+j)=Mat[imat][jmat]+(*k_val)*(Mat[imat+1][jmat]+Mat[imat-1][jmat]+Mat[imat][jmat+1]+Mat[imat][jmat-1]-4*Mat[imat][jmat]);
			
				for (int ii = 0; ii < *pn; ii+=5)
				{
					if (j>=(int)sp[ii]&&i>=(int)sp[ii+1]&&j<((int)sp[ii]+(int)sp[ii+2])&&i<((int)sp[ii+1]+(int)sp[ii+3]))
					{
						*(Tnew+i*(*grid_h)+j)=(float)sp[ii+4];
					}
				}
			*(Told+i*(*grid_h)+j)=*(Tnew+i*(*grid_h)+j);
			}
		}

	}
	else if (*dim ==3)
	{   
		__shared__ float Mat[BLOCK_SIZE+2][BLOCK_SIZE+2][BLOCK_SIZE+2];
		int j = blockIdx.x * blockDim.x + threadIdx.x;
		int i = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		int jmat = threadIdx.x+1;
		int imat = threadIdx.y+1;
		int kmat = threadIdx.z+1;

		if (i<=*grid_h && j<=*grid_w && k<=*grid_d)
		{
			int lengthj=(blockIdx.x==(int)*grid_w/BLOCK_SIZE)? *grid_w%BLOCK_SIZE :BLOCK_SIZE;
			int lengthi=(blockIdx.y==(int)*grid_h/BLOCK_SIZE)? *grid_h%BLOCK_SIZE :BLOCK_SIZE;
			int lengthk=(blockIdx.z==(int)*grid_d/BLOCK_SIZE)? *grid_d%BLOCK_SIZE :BLOCK_SIZE;
			Mat[imat][jmat][kmat]=Told[i*(*grid_h)*(*grid_d)+j*(*grid_d)+k];

			if (threadIdx.x < 1)
			{
				Mat[imat][jmat-1][kmat]=(j<1)?  *(Told+i*(*grid_h)*(*grid_d)+j*(*grid_d)+k): *(Told+i*(*grid_h)*(*grid_d)+(j-1)*(*grid_d)+k);  /// check j or j-1
				Mat[imat][jmat + lengthj][kmat]=(j >= *grid_w-lengthj)? *(Told+i*(*grid_h)*(*grid_d)+(j+lengthj-1)*(*grid_d)+k):*(Told+i*(*grid_h)*(*grid_d)+(j+lengthj)*(*grid_d)+k);
			}

			 if (threadIdx.y < 1)
			{
				Mat[imat-1][jmat][kmat]= (i < 1)? *(Told+i*(*grid_h)*(*grid_d)+j*(*grid_d)+k): *(Told+(i-1)*(*grid_h)*(*grid_d)+j*(*grid_d)+k);  /// check j or j-1
				Mat[imat + lengthi][jmat][kmat]=(i >= *grid_h-lengthi)? *(Told+(i+lengthi-1)*(*grid_h)*(*grid_d)+j*(*grid_d)+k): *(Told+(i+lengthi)*(*grid_h)*(*grid_d)+j*(*grid_d)+k);	
				
			}
			 if (threadIdx.z < 1)
			{
				Mat[imat][jmat][kmat-1]= (k < 1)? *(Told+i*(*grid_h)*(*grid_d)+j*(*grid_d)+k): *(Told+i*(*grid_h)*(*grid_d)+j*(*grid_d)+k-1);  /// check j or j-1
				Mat[imat][jmat][kmat+lengthk]= (k >= *grid_d-lengthk)? *(Told+i*(*grid_h)*(*grid_d)+j*(*grid_d)+k+lengthk-1): *(Told+i*(*grid_h)*(*grid_d)+j*(*grid_d)+k+lengthk);	
			}
			__syncthreads();
			
			if (i<*grid_h && j<*grid_w && k<*grid_d)
			{
				*(Tnew+i*(*grid_h)*(*grid_d)+j*(*grid_d)+k)=Mat[imat][jmat][kmat]+(*k_val)*(Mat[imat+1][jmat][kmat]+Mat[imat-1][jmat][kmat]+Mat[imat][jmat+1][kmat]+Mat[imat][jmat-1][kmat]+Mat[imat][jmat][kmat-1]+Mat[imat][jmat][kmat+1]-6*Mat[imat][jmat][kmat]);
			
				for (int ii = 0; ii < *pn; ii+=7)
				{
					if (j>=(int)sp[ii]&&i>=(int)sp[ii+1]&&k>=(int)sp[ii+2]&&j<((int)sp[ii]+(int)sp[ii+3])&&i<((int)sp[ii+1]+(int)sp[ii+4])&&k<((int)sp[ii+2]+(int)sp[ii+5]))
					{
						*(Tnew+i*(*grid_h)*(*grid_d)+j*(*grid_d)+k)=(float)sp[ii+6];
					}
				}
			*(Told+i*(*grid_h)*(*grid_d)+j*(*grid_d)+k)=*(Tnew+i*(*grid_h)*(*grid_d)+j*(*grid_d)+k);
			}
		}
	}
}

int main(int argc, char const *argv[])
{

	if (argc!=2)
	{
		cout<<"There seems to be a problem with number of arguments!"<<endl;
		exit(1);
	}
	// parameters
	vector <string> pstring;
	// Parsing the conf file values
	ifstream input_file(argv[1]);
	string Line;
	int paramter_number=0;
	string temp;
	int size=0;
	while(getline(input_file,Line))
	{
		if (Line.length()==0)
		{
			continue;
		}
			for (int i = 0; i < Line.length() ; i++)
			{
				if (Line[i]==' '||(int)Line[i]==13||(int)Line[i]==9)
				{
					continue;	
				}
				else
				{
					if (Line.at(i)=='#')
					{
						break;
					}
					if (Line.at(i)==',')
					{
						pstring.push_back(temp);
						temp.clear();

						size=0;
						paramter_number+=1;
						continue;
					}
					temp.push_back(Line.at(i));	
					size+=1;						
				}
			}
			if (temp.length()!=0)
			{
			pstring.push_back(temp);
			temp.clear();
			size=0;
			paramter_number+=1;
			}
	}
	input_file.close();

	// Getting all values
	int dimension=(int)(pstring[0].at(0))-48;
	pstring[1].insert(0,1,'0');
	float k=atof(pstring[1].c_str());
	int timestep = atoi(pstring[2].c_str());
	std::vector<float> source_parameters;

	int grid_width,grid_height,grid_depth;
	float start_temp;
	/*cout<<"\nNumber of Dimensions : "<<dimension;
	cout<<"\nK                    : "<<k;
	cout<<"\nTimestep             : "<<timestep;*/
	float *d_Told,*d_Tnew,*d_k_val,*d_sp;
	int *d_dim,*d_grid_w,*d_grid_h,*d_grid_d,*pn;

	if (dimension==2)
	{
		grid_width= atoi(pstring[3].c_str());
		grid_height=atoi(pstring[4].c_str());
		start_temp=atof(pstring[5].c_str());
		/*cout<<"\nGrid Width           : "<<grid_width;
		cout<<"\nGrid Height          : "<<grid_height;
		cout<<"\nStarting Temperature : "<<start_temp<<endl;*/
		for (int i = 6; i < paramter_number; i++)
		{
			source_parameters.push_back(atof(pstring[i].c_str()));
		}
		/*for (int i = 0; i < paramter_number-6; i=i+5)
		{
			cout<<"\nX : "<<source_parameters[i]<<" Y : "<<source_parameters[i+1];
			cout<<" Width : "<<source_parameters[i+2]<<" Height : "<<source_parameters[i+3];
			cout<<" Temperature : "<<source_parameters[i+4]<<endl;
		}*/
		int length = (grid_width>=grid_height)? grid_width*grid_width:grid_height*grid_height;
		float Told[length] = {0};
		float Tnew[length] = {0}; 
		int size_T = (length)*sizeof(float);

		for (int i = 0; i < grid_height; i++)
		{
			for (int j = 0; j < grid_width; j++)
			{
				for (int ii = 0; ii < (paramter_number-6); ii+=5)
				{
					if (j>=(int)source_parameters[ii]&&i>=(int)source_parameters[ii+1]&&j<((int)source_parameters[ii]+(int)source_parameters[ii+2])&&i<((int)source_parameters[ii+1]+(int)source_parameters[ii+3]))
					{
						Told[i*grid_height+j]=source_parameters[ii+4];
						break;
					}
					else
					{
						Told[i*grid_height+j]=start_temp;
					}
				}
			}
		}
		int p_n=paramter_number-6;
		float* sp = &source_parameters[0];

		cudaMalloc((void **)&d_Told, size_T);
		cudaMalloc((void **)&d_Tnew, size_T);
		cudaMalloc((void **)&d_k_val,sizeof(float));
		cudaMalloc((void **)&d_dim,sizeof(int));
		cudaMalloc((void **)&d_sp,p_n*sizeof(float));
		cudaMalloc((void **)&d_grid_w,sizeof(int));
		cudaMalloc((void **)&d_grid_h,sizeof(int));
		cudaMalloc((void **)&d_grid_d,sizeof(int));
		cudaMalloc((void **)&pn,sizeof(int));

		cudaMemcpy(d_Told,Told,size_T,cudaMemcpyHostToDevice);
		cudaMemcpy(d_Tnew,Tnew,size_T,cudaMemcpyHostToDevice);
		cudaMemcpy(d_k_val,&k,sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(d_dim,&dimension,sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(d_sp,sp,p_n*sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(d_grid_w,&grid_width,sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(d_grid_h,&grid_height,sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(d_grid_d,&grid_depth,sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(pn,&p_n,sizeof(int),cudaMemcpyHostToDevice);

		dim3 Block(BLOCK_SIZE,BLOCK_SIZE);
		dim3 Grid((int)ceil((grid_width+BLOCK_SIZE-1)/BLOCK_SIZE),(int)ceil((grid_height+BLOCK_SIZE-1)/BLOCK_SIZE));

		for (int t = 0; t <timestep; t++)
		{	
			kernel<<<Grid,Block>>>(d_Told,d_Tnew,d_dim,d_k_val,d_sp,d_grid_w,d_grid_h,d_grid_d,pn);
		}
		cudaMemcpy(Tnew,d_Tnew,size_T,cudaMemcpyDeviceToHost);
		cudaFree(d_Tnew);cudaFree(d_Told);cudaFree(d_k_val);cudaFree(d_dim);cudaFree(d_sp);cudaFree(d_sp);
		cudaFree(d_grid_d);cudaFree(d_grid_h);cudaFree(d_grid_w);
		/*cout<<"\nTnew: \n";
		for (int i = 0; i < grid_height; i++)
		{
			for (int j = 0; j < grid_width-1; j++)
			{
				cout<<left<<Tnew[i*grid_height+ j]<<", ";
				//printf("%f, ",Tnew[i][j] );
			}
			cout<<left<<Tnew[i*grid_height + grid_width-1]<<endl;
			//printf("%f,\n",Tnew[i][grid_width-1] );
		}*/
		ofstream build ("heatOutput.csv", std::ofstream::out);
		for (int i = 0; i < grid_height; i++)
		{
			for (int j = 0; j < grid_width-1; j++)
			{
				build<<std::setprecision(6)<<Tnew[i*grid_height+ j]<<", ";
			}
			build<<std::setprecision(6)<<Tnew[i*grid_height+ grid_width-1];
			if (i!=grid_height-1)
			{
				build<<"\n";
			}
		}
		build.close();
	}
	else if (dimension==3)
	{
		grid_width= atoi(pstring[3].c_str());
		grid_height=atoi(pstring[4].c_str());
		grid_depth=atoi(pstring[5].c_str());
		start_temp=atof(pstring[6].c_str());
		/*cout<<"\nGrid Width           : "<<grid_width;
		cout<<"\nGrid Height          : "<<grid_height;
		cout<<"\nGrid Depth           : "<<grid_depth;
		cout<<"\nStarting Temperature : "<<start_temp<<endl;*/
		for (int i = 7; i < paramter_number; i++)
		{
			source_parameters.push_back(atof(pstring[i].c_str()));
		}
		/*for (int i = 0; i < paramter_number-7; i=i+7)
		{
			cout<<"\nX : "<<source_parameters[i]<<" Y : "<<source_parameters[i+1]<<" Z : "<<source_parameters[i+2] ;
			cout<<" Width : "<<source_parameters[i+3]<<" Height : "<<source_parameters[i+4]<< " Depth : "<<source_parameters[i+5];
			cout<<" Temperature: "<<source_parameters[i+6]<<endl;
		}*/
		int length = (grid_width>=grid_height)?((grid_width>=grid_depth)?grid_width*grid_width*grid_width:grid_depth*grid_depth*grid_depth):((grid_height>=grid_depth)?grid_height*grid_height*grid_height:grid_depth*grid_depth*grid_depth);
		float Told[length] = {0};
		float Tnew[length] = {0}; 
		int size_T = (length)*sizeof(float);

		for (int kk = 0; kk < grid_depth; kk++)
			{
			for (int i = 0; i < grid_height; i++)
				{
					for (int j = 0; j < grid_width; j++)
					{
						for (int ii = 0; ii < (paramter_number-7); ii+=7)
						{
							if (j>=(int)source_parameters[ii]&&i>=(int)source_parameters[ii+1]&&kk>=(int)source_parameters[ii+2]&&j<((int)source_parameters[ii]+(int)source_parameters[ii+3])&&i<((int)source_parameters[ii+1]+(int)source_parameters[ii+4])&&kk<((int)source_parameters[ii+2]+(int)source_parameters[ii+5]))
							{
								Told[i*(grid_height)*(grid_depth)+j*(grid_depth)+kk]=source_parameters[ii+6];
								break;
							}
							else
							{
								Told[i*(grid_height)*(grid_depth)+j*(grid_depth)+kk]=start_temp;
							}
						}
					}
				}
			}

		int p_n=paramter_number-7;
		float* sp = &source_parameters[0];

		cudaMalloc((void **)&d_Told, size_T);
		cudaMalloc((void **)&d_Tnew, size_T);
		cudaMalloc((void **)&d_k_val,sizeof(float));
		cudaMalloc((void **)&d_dim,sizeof(int));
		cudaMalloc((void **)&d_sp,p_n*sizeof(float));
		cudaMalloc((void **)&d_grid_w,sizeof(int));
		cudaMalloc((void **)&d_grid_h,sizeof(int));
		cudaMalloc((void **)&d_grid_d,sizeof(int));
		cudaMalloc((void **)&pn,sizeof(int));

		cudaMemcpy(d_Told,Told,size_T,cudaMemcpyHostToDevice);
		cudaMemcpy(d_Tnew,Tnew,size_T,cudaMemcpyHostToDevice);
		cudaMemcpy(d_k_val,&k,sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(d_dim,&dimension,sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(d_sp,sp,p_n*sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(d_grid_w,&grid_width,sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(d_grid_h,&grid_height,sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(d_grid_d,&grid_depth,sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(pn,&p_n,sizeof(int),cudaMemcpyHostToDevice);

		dim3 Block(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE);
		dim3 Grid((int)ceil((grid_width+BLOCK_SIZE-1)/BLOCK_SIZE),(int)ceil((grid_height+BLOCK_SIZE-1)/BLOCK_SIZE),(int)ceil((grid_depth+BLOCK_SIZE-1)/BLOCK_SIZE));

		for (int t = 0; t <timestep; t++)
		{	
			kernel<<<Grid,Block>>>(d_Told,d_Tnew,d_dim,d_k_val,d_sp,d_grid_w,d_grid_h,d_grid_d,pn);
		}
		cudaMemcpy(Tnew,d_Tnew,size_T,cudaMemcpyDeviceToHost);
		cudaFree(d_Tnew);cudaFree(d_Told);cudaFree(d_k_val);cudaFree(d_dim);cudaFree(d_sp);cudaFree(d_sp);
		cudaFree(d_grid_d);cudaFree(d_grid_h);cudaFree(d_grid_w);

		/*cout<<"\nTnew: \n";
		for (int kk = 0; kk < grid_depth ; kk++)
		{
			cout<<"\n";
			for (int i = 0; i < grid_height; i++)
			{
				for (int j = 0; j < grid_width-1; j++)
				{
					cout<<left<<Tnew[i*(grid_height)*(grid_depth)+j*(grid_depth)+kk]<<", ";
					//printf("%f, ",Tnew[i][j] );
				}
				cout<<left<<Tnew[(i*grid_height + grid_width-1)*grid_depth+kk]<<endl;
				//printf("%f,\n",Tnew[i][grid_width-1] );
			}
			cout<<"\n";
		}*/
		ofstream build ("heatOutput.csv", std::ofstream::out);
		for (int kk = 0; kk < grid_depth; kk++)
		{
			for (int i = 0; i < grid_height; i++)
			{
				for (int j = 0; j < grid_width-1; j++)
				{
					build<<std::setprecision(6)<<Tnew[i*(grid_height)*(grid_depth)+j*(grid_depth)+kk]<<", ";
				}
				build<<std::setprecision(6)<<Tnew[i*(grid_height)*(grid_depth)+(grid_width-1)*(grid_depth)+kk];
				if (kk!=grid_depth-1||i!=grid_height-1)
				{	
					build<<"\n";
				}
			}
			if (kk!=grid_depth-1)
			{
				build<<"\n";
			}
			
		}
		build.close();
	}
	return 0;
}