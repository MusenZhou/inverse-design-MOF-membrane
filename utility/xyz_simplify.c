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
	fp2 = fopen(argv[2], "w+");
	fscanf(fp1, " %d\n", &N_original);
	fgets(str, buffersize, fp1);
	// count how many connections are recorded in the comment line of xyz file
	for (i=0; i<buffersize; i++)
	{
		if (isdigit(str[i]) && isspace(str[i+1]))
		{
			N_connection++;
		}
		else if (isdigit(str[i]) && (str[i+1]=='\n'))
		{
			N_connection++;
		}
		else if (str[i]=='\n')
		{
			break;
		}
	}
	printf("%d\n", N_connection);
	N_actual = N_original - N_connection;
	fprintf(fp2, "%d\n", N_actual);
	fprintf(fp2, "\n");

	for (i=0; i<N_actual; i++)
	{
		fgets(str, buffersize, fp1);
		fprintf(fp2, "%s", str);
	}
	fclose(fp1);
	fclose(fp2);
}
