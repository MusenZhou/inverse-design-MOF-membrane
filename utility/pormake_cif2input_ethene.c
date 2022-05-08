#include <stdio.h>
#include <string.h>


int main(int argc, char *argv[])
{
	//This script needs three input string as following: cif file, force field info file and output file name
	//***********
	//This part defines the input parameter
	// double dL = DL;
	double dL = 1;
	int Nmax_alpha = 8;
	int Nmax_beta = 4;
	int Nmax_gamma = 8;
	int FH_signal = 0;
	double Temperature = 300;
	double cutoff = 12.9;
	//This part defines other parameter needed to write the input file
	int Nmaxa, Nmaxb, Nmaxc;
	//*************



	//define file varaiable
	FILE *fp1, *fp2;
	int buffersize = 256;
	char str[buffersize];
	//define read-in parameters
	double alpha = 0, beta = 0, gamma = 0;
	double a, b, c;
	int N;
	int i, ii, j;
	double occupancy;
	double cartesian_coordinate[3];
	char atom[2];
	double mass;

	char str0[] = "loop_";
	char str1[] = "_cell_length_a";
	char str2[] = "_cell_length_b";
	char str3[] = "_cell_length_c";
	char str4[] = "_cell_angle_alpha";
	char str5[] = "_cell_angle_beta";
	char str6[] = "_cell_angle_gamma";
	char str7[] = "_atom_site_type_symbol";
	char str8[] = "_atom_site_occupancy";
	char str9[] = "_atom_type_partial_charge";
	char empty_line[] = "\n";
	char space[] = " ";
	double x, y, z;
	//done!!!!!

	//parameter read
	//e.g. lattice size and angle
	fp1 = fopen(argv[1], "r");
	while (1)
	{
		fscanf(fp1, "%s", str);
		if (strcmp(str, str1) == 0)
		{
			fscanf(fp1, "%lf", &a);
		}
		else if (strcmp(str, str2) == 0)
		{
			fscanf(fp1, "%lf", &b);
		}
		else if (strcmp(str, str3) == 0)
		{
			fscanf(fp1, "%lf", &c);
		}
		else if (strcmp(str, str4) == 0)
		{
			fscanf(fp1, "%lf", &alpha);
		}
		else if (strcmp(str, str5) == 0)
		{
			fscanf(fp1, "%lf", &beta);
		}
		else if (strcmp(str, str6) == 0)
		{
			fscanf(fp1, "%lf", &gamma);
		}
		else if ( fgets(str, buffersize, fp1) == NULL)
		{
			break;
		}
	}
	fclose(fp1);
	// printf("%lf %lf %lf %lf %lf %lf\n", a, b, c, alpha, beta, gamma);
	//done!!!!!



	// figure out how many line to skip before atomistic information
	int sss1=0, sss2=0;
	int good_signal1 = 0, good_signal2 = 0;
	char extract[buffersize];
	int continue_space;
	int det1;
	fp1 = fopen(argv[1], "r");
	while (1)
	{
		fscanf(fp1, "%s", str);
		if (strcmp(str, str9) == 0)
		{
			sss1++;
			break;
		}
		else
		{
			sss1++;
			if ( fgets(str, buffersize, fp1) == NULL)
			{
				break;
			}

		}
	}
	// printf("N1: %d\n", sss1);
	fclose(fp1);
	fp1 = fopen(argv[1], "r");
	for (i=0; i<sss1; i++)
	{
		fgets(str, buffersize, fp1);
	}
	// printf("%s", str);
	while (1)
	{
		fscanf(fp1, "%s", str);
		if (strcmp(str, str0) == 0)
		{
			break;
		}
		else
		{
			sss2++;
			if ( fgets(str, buffersize, fp1) == NULL)
			{
				break;
			}

		}
	}
	// printf("N2: %d\n", sss2);
	fclose(fp1);
	int skip_line;
	skip_line = sss1;
	// done!!!!!



	// count atom number
	N = sss2;
	// done!!!

	//store forcefield information before write input file
	//count atom number from forcefiled file
	fp1 = fopen(argv[2], "r");
	fgets(str, buffersize, fp1);
	int N_P = 0;
	while (1)
	{
		if ( fgets(str, buffersize, fp1) != NULL)
		{
			N_P++;
		}
		else
		{
			fclose(fp1);
			break;
		}
	}
	//read atomistic signature from forcefiled file
	char atom_list[N_P][3];
	double signature_list[N_P*3];
	fp1 = fopen(argv[2], "r");
	fgets(str, buffersize, fp1);
	for (i=0; i<N_P; i++)
	{
		// read atom name, sigma, epsilon and mass from forcefield file
		fscanf(fp1, "%s %lf %lf %lf\n", atom_list[i], &signature_list[3*i], &signature_list[3*i+1], &signature_list[3*i+2]);
	}
	fclose(fp1);
	//done!!!

	// write input file
	fp1 = fopen(argv[1], "r");
	fp2 = fopen(argv[3], "w+");
	// write title part
	fprintf(fp2,"Nmaxa Nmaxb Nmaxc:\n");
	Nmaxa = a/dL + 3;
	Nmaxb = b/dL + 3;
	Nmaxc = c/dL + 3;
	fprintf(fp2,"%d\t%d\t%d\n", Nmaxa, Nmaxb, Nmaxc);
	fprintf(fp2,"La Lb Lc dL\n");
	fprintf(fp2,"%lf\t%lf\t%lf\t%lf\n", a, b, c, dL);
	fprintf(fp2,"Alpha Beta Gamma\n");
	fprintf(fp2,"%lf\t%lf\t%lf\n", alpha, beta, gamma);
	fprintf(fp2,"N_angle_alpha N_angle_beta N_angle_gamma\n");
	fprintf(fp2,"%d\t%d\t%d\n", Nmax_alpha, Nmax_beta, Nmax_gamma);
	fprintf(fp2,"cutoff(A) FH_signal mass(g/mol) Tempearture(K)\n");
	fprintf(fp2,"%lf\t%d\t28\t%lf\n", cutoff, FH_signal, Temperature);
	fprintf(fp2,"------------------Adsorbate------------------\n");
	fprintf(fp2,"Number of sites\n");
	fprintf(fp2,"2\n");
	fprintf(fp2,"x(A)	y(A)	z(A)	Epsilon(K)	Sigma(A)	Mass(g/mol)\n");
	fprintf(fp2,"0.00 0.00 0.00 92.8 3.68 14\n");
	fprintf(fp2,"0.00 0.00 1.33 92.8 3.68 14\n");	
	fprintf(fp2,"------------------Adsorbent------------------\n");
	fprintf(fp2,"Number of atoms\n");
	fprintf(fp2,"%d\n", N);
	fprintf(fp2,"ID diameter(A) Epsilon(K) mass(g/mol) frac_x frac_y frac_z atom_name\n");
	for (i=0; i<skip_line; i++)
	{
		// skip lines
		fgets(str, buffersize, fp1);
		// done!!!
	}
	for (i=0; i<N; i++)
	{
		fscanf(fp1,"%s ", str);
		fscanf(fp1,"%s", atom);
		fscanf(fp1,"%lf %lf %lf", &x, &y, &z);
		fgets(str, buffersize, fp1);
		// done!!!
		if (atom[1]=='\0')
		{
			// element name has only one character
			for (ii=0; ii<N_P; ii++)
			{
				if (atom_list[ii][1]=='\0')
				{
					if (atom_list[ii][0]==atom[0])
					{
						break;
					}
				}
			}
		}
		else
		{
			// element name has two character
			for (ii=0; ii<N_P; ii++)
			{
				if ((atom_list[ii][0]==atom[0])&&(atom_list[ii][1]==atom[1]))
				{
					break;
				}
			}
		}
		fprintf(fp2,"%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%s\n", i+1, signature_list[3*ii], signature_list[3*ii+1], signature_list[3*ii+2], x, y, z, atom_list[ii]);
	}
	fclose(fp1);
	fclose(fp2);
}
