#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<ctype.h>

int main(int argc, char *argv[])
{
	int buffersize = 512;
	char str[buffersize];
	int i;



	int N_original;
	int N_connection=0;
	int N_actual;



	FILE *fp1, *fp2;



	fp1 = fopen(argv[1], "r");
	fscanf(fp1, " %d\n", &N_original);
	// count how many connections (Ar) are in the xyz file
	for (i=0; i<N_original; i++)
	{
		fgets(str, buffersize, fp1);

		if ((str[0]=='A')&&(str[1]=='r'))
		{
			N_connection++;
		}
	}
	fclose(fp1);
	// printf("%d\n", N_connection);
	N_actual = N_original - N_connection;
	if (N_actual>0)
	{
		fp2 = fopen(argv[2], "w+");
		fprintf(fp2, "%d\n", N_actual);
		fprintf(fp2, "\n");
		fp1 = fopen(argv[1], "r");
		fscanf(fp1, " %d\n", &N_original);
		for (i=0; i<N_actual; i++)
		{
			fgets(str, buffersize, fp1);
			fprintf(fp2, "%s", str);
		}
		fclose(fp1);
		fclose(fp2);
	}
	
}
