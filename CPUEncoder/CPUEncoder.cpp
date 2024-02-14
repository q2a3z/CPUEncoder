#include "stdafx.h"

#include <cv.h>
#include <highgui.h>
#define OUTPUT_TIME 0
#define OUTPUT_IMG 0
#define OUTPUT_COUNT 0
#define OUTPUT_DATA 0
#define OUTPUT_INT_DATA 0
#define OUTPUT_UCHAR_DATA 0
#define OUTPUT_ENTROPY 0
#define OUTPUT_BITRATE 0
#define ERROR_CONTEXT 0
#define OUTPUT_CONTEXT_SD 0
#define SIGN_FILP 0
#define REMAPPING 0
#define ADAPTIVE_ARITHMETIC_CODING 0
#define ADAPTIVE_GOLOMB_CODING 1
#define TEST_ONE 1
#define pureMED 0
#define AllLSR 0
#define SettingN 1
#define contextLSR 1
#define PredictorOrderSetting 6
#define EdgeDetectCoffNW 12
#define EdgeDetectCoffEw 8

#define Pixel_Shift 128
#define ContextArraySize 1024

#define FILE_PATH "C:\\img\\"
#define FILE_OUT_PATH "C:\\img\\out\\"
#define Pixel_Byte 1
//Traning Area
#define traning_area_left  6				//_____7*6_______6*7____
#define traning_area_right 6			//|			         |		              |
#define traning_area_top 6				//|			         |			          |
#define traning_area_last 6				//|			         |__________|
#define traning_area_size 84			//|__________|x

//#define MAX(a,b) ( ( a > b) ? a : b )
//#define MIN(a,b) ( ( a < b) ? a : b )

//**********************Least Square Resolve****************************//
#define NR_END 1
#define FREE_ARG char*
#define MAX_nn_size 15 //cholesky

static float sqrarg;
#define SQR(a) ((sqrarg=(a)) == 0.0 ? 0.0 : sqrarg*sqrarg)
static float maxarg1, maxarg2;
#define FMAX(a,b) (maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ?\
(maxarg1) : (maxarg2))

#define FMIN(a,b) (minarg1=(a),minarg2=(b),(minarg1) < (minarg2) ?\
(minarg1) : (minarg2))

#define LMAX(a,b) (lmaxarg1=(a),lmaxarg2=(b),(lmaxarg1) > (lmaxarg2) ?\
(lmaxarg1) : (lmaxarg2))

#define LMIN(a,b) (lminarg1=(a),lminarg2=(b),(lminarg1) < (lminarg2) ?\
(lminarg1) : (lminarg2))

#define IMAX(a,b) (imaxarg1=(a),imaxarg2=(b),(imaxarg1) > (imaxarg2) ?\
(imaxarg1) : (imaxarg2))
static int iminarg1, iminarg2;
#define IMIN(a,b) (iminarg1=(a),iminarg2=(b),(iminarg1) < (iminarg2) ?\
(iminarg1) : (iminarg2))
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
//**********************Least Square Resolve****************************//

//------------------Arithmetic Coding-------------------------------//
#define NUMLOOPS 10000
#define ADAPT 1
#define FILE1 "foo"
#define MASK1 ((0x1<<3) - 1)
#define NSYM1 (MASK1 + 1)

#define Code_value_bits 16
#define Top_value (((long)1<<Code_value_bits)-1)
#define First_qtr (Top_value/4+1)
#define Half	  (2*First_qtr)
#define Third_qtr (3*First_qtr)
#define Max_frequency 16383

typedef struct {
	FILE *fp;
	long low;
	long high;
	long fbits;
	int buffer;
	int bits_to_go;
	long total_bits;
} ac_encoder;

typedef struct {
	FILE *fp;
	long value;
	long low;
	long high;
	int buffer;
	int bits_to_go;
	int garbage_bits;
} ac_decoder;

typedef struct {
	int nsym;
	int *freq;
	int *cfreq;
	int adapt;
} ac_model;
//----------------------------Arithmetic Coding-------------------------------//
//----------------------------Golomb Coding-------------------------------//



//----------------------------Golomb Coding-------------------------------//

#define LN2 0.693147 //ln2
typedef struct {
	FILE *fp;
	int buffer;
	int bits_to_go;
	long total_bits;
} ag_encoder;

typedef struct {
	FILE *fp;
	long value;
	int buffer;
	int bits_to_go;
	int garbage_bits;
} ag_decoder;

typedef struct {
	int nsym;
	int asym;
	int k;
	int adapt;
} ag_model;

//--------------------------Entropy--------------------------------//
#define MAXLEN 100 //maximum string length
#define HISSIZE 513 //maximum string length
double Log2(double n)
{
	// log(n)/log(2) is log2.  
	return log(n) / log(2);
}
int makehist(int *S, int *hist, int len){
	int wherechar[HISSIZE];
	int i, histlen;
	histlen = 0;
	for (i = 0; i<HISSIZE; i++)wherechar[i] = -1;
	for (i = 0; i<len; i++){
		if (wherechar[(int)(S[i] + 256)] == -1){
			wherechar[(int)(S[i] + 256)] = histlen;
			histlen++;
		}
		hist[wherechar[(int)(S[i] + 256)]]++;
	}
	return histlen;
}
double entropy(int *hist, int histlen, int len){
	int i;
	double H;
	H = 0;
	for (i = 0; i<histlen; i++){
		H -= (double)hist[i] / len*Log2((double)hist[i] / len);
	}
	return H;
}
//--------------------------Entropy--------------------------------//

using namespace cv;

unsigned int Height, Width;
int x, n, w, ne, nw, nn, ww, nne;//Neiber Pixel
int predicted; //預測影像值xp
int predictor_order;//預測階數
float a16[12] = {0.1666666 };//33311
int delta = 0;
int Count = 0;

//double q[8] = { 1, 2, 3, 4, 6, 9, 14, 256 }; //delta = |e|
//int q[8] = { 2, 5, 12, 104, 160, 169, 188, 256 }; //delta = MED |e|
double q[8] = { 5, 10, 15, 25, 42, 60, 85, 140 };//EDP
//double q[8] = { 9, 16, 29, 45, 73, 121, 222, 255 };//Re EDP 6p
//double q[8] = { 2, 5, 7, 9, 72, 118, 222, 255 };//Re EDP 7p
//double q[8] = { 1, 11, 31, 51, 81, 121, 221, 255 };//MED*10
//int q[8] = { 3, 14, 59, 256 };
//int q[8] = { 3, 5, 8,14, 59, 256 };
//int N[8] = { 5, 8, 14, 59, 128, 256 };
//int N[8] = { 39, 79, 134, 256 };

//int N[8] = { 15, 20, 35, 50, 80, 130, 250, 256};
//int N[8] = { 2, 11, 31, 51, 81, 121, 221, 256 };
//int N[8] = { 256, 256, 256,256, 256, 256, 256, 256 };
//int N[8] = { 29, 10, 16, 27, 38, 58, 97, 256 };//Re EDP N
//int N[8] = { 29, 38, 58, 97, 147,256 };//Re EDP N
int N[8] = { 7, 11, 20, 24, 34, 46, 103, 256 };//EDP N
//int q[8] =  { 1,2, 3, 5, 8, 14, 59, 256 };
/*
int N[8] = { 19, 27, 39, 54, 62, 81, 106, 128 };
int nN[8] = { 16, 23, 37, 52, 66, 85, 108, 128 };
int q[8] = { 3, 8, 16, 30, 44, 82, 237, 256 };
*/
int Count1 = 0, Count2 = 0, Count3 = 0
, Count4 = 0, Count5 = 0, Count6 = 0
, Count7 = 0, Count8 = 0, Count9 = 0;

int  Se[ContextArraySize] = { 0 }, Sa[ContextArraySize] = { 0 }, context_count[ContextArraySize] = { 0 };

int times_count[ContextArraySize] = { 0 }, context_val[ContextArraySize][1024 * 100] = { 0 };
float context_avg[ContextArraySize] = { 0 }, context_var[ContextArraySize] = { 0 };

char *FILEName;

unsigned int choleskyCount = 0, svdCount = 0;//Cholesky & SVD Counter
double compress_time;
double normal_time = 0, TranArea_time = 0, TranSize_time = 0;
clock_t compress_start, compress_start2, compress_end;
clock_t TranSize_start, TranSize_end, TranArea_start, TranArea_end, normal_start, normal_end;

ac_encoder ace1;
ac_model acm1, acm2, acm3, acm4, acm5, acm6, acm7, acm8,acm9;
ag_encoder age1;
ag_decoder agd1;
ag_model agm1, agm2, agm3, agm4, agm5, agm6, agm7, agm8,agm9;

/******************************************程式開始*********************************************/

//3 String Combin
char *FileName(char *filepath, char *filename, char *filetype) {
	// 計算所需的陣列長度  
	int length = strlen(filepath) + strlen(filename) + strlen(filetype) + 1;
	// 產生新的陣列空間  
	char *result = (char*)malloc(sizeof(char) * length);
	// 複製第一個字串至新的陣列空間  
	strcpy(result, filepath);
	// 串接第二個字串至新的陣列空間  
	strcat(result, filename);
	strcat(result, filetype);
	return result;
}
//4 String Combin
char *OutFileName(char *filepath, char *filename, char *times, char *filetype) {
	// 計算所需的陣列長度  
	int length = strlen(filepath) + strlen(filename) + strlen(times) + strlen(filetype) + 1;
	// 產生新的陣列空間  
	char *result = (char*)malloc(sizeof(char) * length);
	// 複製第一個字串至新的陣列空間  
	strcpy(result, filepath);
	// 串接第二個字串至新的陣列空間  
	strcat(result, filename);
	strcat(result, times);
	strcat(result, filetype);
	return result;
}
//ReadImage OpenCV RawData -> Source (unsigned char -> int)
void ReadImage(unsigned char* rawdata, int* SourceImage){
	for (unsigned int w = 0; w < Width; w++){
		for (unsigned int h = 0; h < Height; h++){
			unsigned int x = w * Height + h;
			*(SourceImage + x) = *(rawdata + x);
		}
	}
}
//Write Image Source -> OpenCV RawData (int ->unsigned char)
void WriteImage(int* OutImage, unsigned char* Outrawdata){
	for (unsigned int w = 0; w < Width; w++){
		for (unsigned int h = 0; h < Height; h++){
			unsigned int x = w * Height + h;
			//if (*(OutImage + x) > 128 || *(OutImage + x) < -128)
			//	*(Outrawdata + x) = Pixel_Shift;
			//else
			*(Outrawdata + x) = *(OutImage + x) + Pixel_Shift;
		}
	}
}
//WriteData To Txt
void writedata(char *data, char *filename){
	FILE *fp;
	char *c;
	c = (char *)malloc(sizeof(char) * 2);
	sprintf(c, "%d", predictor_order);
	if ((fp = fopen(OutFileName(FILE_OUT_PATH, filename, c, ".txt"), "w")) == 0){
		printf("Write Fail\n");
		system("pause");
	}
	else{
		//fwrite(header, sizeof(unsigned char), 1078, fp);
		//fseek(fp, 1078, SEEK_SET);
		fwrite(data, sizeof(char) * 8, (Width)*(Height)*(Pixel_Byte), fp);
		fclose(fp);
		printf("WRITE_FILE...OK!\n");
	}
}
//WriteTime
void writetime(char *data, char *filename){
	FILE *fp;
	if ((fp = fopen(FileName(FILE_OUT_PATH, filename, "time.txt"), "a")) == 0){
		printf("Write Fail\n");
		system("pause");
	}
	else{
		//fwrite(header, sizeof(unsigned char), 1078, fp);
		//fseek(fp, 1078, SEEK_SET);
		fwrite(data, sizeof(char), 30, fp);
		fclose(fp);
		printf("WRITE_FILE...OK!\n");
	}
}
//WriteCount
void writeCounttimes(char *data, char *filename){
	FILE *fp;
	if ((fp = fopen(FileName(FILE_OUT_PATH, filename, "SVD.txt"), "a")) == 0){
		printf("Write Fail\n");
		system("pause");
	}
	else{
		//fwrite(header, sizeof(unsigned char), 1078, fp);
		//fseek(fp, 1078, SEEK_SET);
		fwrite(data, sizeof(char), 15, fp);
		fclose(fp);
		printf("WRITE_FILE...OK!\n");
	}
}
//Write Entropy
void writeEntropy(char *data, char *filename){
	FILE *fp;
	if ((fp = fopen(FileName(FILE_OUT_PATH, filename, "Entropy.txt"), "a")) == 0){
		printf("Write Fail\n");
		system("pause");
	}
	else{
		//fwrite(header, sizeof(unsigned char), 1078, fp);
		//fseek(fp, 1078, SEEK_SET);
		fwrite(data, sizeof(char), 8, fp);
		fclose(fp);
		printf("WRITE_FILE...OK!\n");
	}
}

void writeintdata(int *data, char *filename){
	FILE *fp;
	char *c;
	c = (char *)malloc(sizeof(char) * 2);
	sprintf(c, "%d", predictor_order);
	if ((fp = fopen(OutFileName(FILE_OUT_PATH, filename, c, "Pre.txt"), "w")) == 0){
		printf("Write Fail\n");
		system("pause");
	}
	else{
		for (int i = 0; i<(Width)*(Height)*(Pixel_Byte); i++){
			fprintf(fp, "%d\n", *(data+i));
		}
		//fwrite(header, sizeof(unsigned char), 1078, fp);
		//fseek(fp, 1078, SEEK_SET);
		//fwrite(data, sizeof(int), (Width)*(Height)*(Pixel_Byte), fp);
		fclose(fp);
		printf("WRITE_FILE...OK!\n");
	}
}

void writeUCdata(int *data, char *filename){
	FILE *fp;
	char *c;
	c = (char *)malloc(sizeof(char) * 2);
	sprintf(c, "%d", predictor_order);
	if ((fp = fopen(OutFileName(FILE_OUT_PATH, filename, c, ".uc"), "wb")) == 0){
		printf("Write Fail\n");
		system("pause");
	}
	else{
		putc((Width >> 8), fp);
		putc((Width & 0x00FF), fp);
		putc((Height >> 8), fp);
		putc((Height & 0x00FF), fp);
		//fseek(fp, 1078, SEEK_SET);
		fwrite(data, sizeof(char), (Width)*(Height)*(Pixel_Byte), fp);
		fclose(fp);
		printf("WRITE_FILE...OK!\n");
	}
}
//Write Bit Rate
void writeBitRate(char *data, char *filename, int size){
	FILE *fp;
	if ((fp = fopen(FileName(FILE_OUT_PATH, "", "BitRate.txt"), "a")) == 0){
		printf("Write Fail\n");
		system("pause");
	}
	else{
		//fwrite(header, sizeof(unsigned char), 1078, fp);
		//fseek(fp, 1078, SEEK_SET);
		fwrite(data, sizeof(char), size, fp);
		fclose(fp);
		printf("WRITE_FILE...OK!\n");
	}
}
//WriteContextSD
void writeContextSD(char *data, char *filename, int size){
	FILE *fp;
	char *c;
	c = (char *)malloc(sizeof(char) * 2);
	sprintf(c, "%d", predictor_order);
	if ((fp = fopen(OutFileName(FILE_OUT_PATH, filename, c, "SD.txt"), "a")) == 0){
		printf("Write Fail\n");
		system("pause");
	}
	else{
		//fwrite(header, sizeof(unsigned char), 1078, fp);
		//fseek(fp, 1078, SEEK_SET);
		fwrite(data, sizeof(char), size, fp);
		fclose(fp);
		//printf("WRITE_FILE...OK!\n");
	}
}

//******************************算術編碼***************************************
static void output_bit(ac_encoder *, int);
static void bit_plus_follow(ac_encoder *, int);
static int input_bit(ac_decoder *);
static void update_model(ac_model *, int);

#define error(m)                                           \
do  {                                                      \
  fflush (stdout);                                         \
  fprintf (stderr, "%s:%d: error: ", __FILE__, __LINE__);  \
  fprintf (stderr, m);                                     \
  fprintf (stderr, "\n");                                  \
  exit (1);                                                \
}  while (0)

#define check(b,m)                                         \
do  {                                                      \
  if (b)                                                   \
    error (m);                                             \
}  while (0)

static void
output_bit(ac_encoder *ace, int bit)
{
	ace->buffer >>= 1;
	if (bit)
		ace->buffer |= 0x80;
	ace->bits_to_go -= 1;
	ace->total_bits += 1;
	if (ace->bits_to_go == 0)  {
		if (ace->fp)
			putc(ace->buffer, ace->fp);
		ace->bits_to_go = 8;
	}
	return;
}

static void
bit_plus_follow(ac_encoder *ace, int bit)
{
	output_bit(ace, bit);
	while (ace->fbits > 0)  {
		output_bit(ace, !bit);
		ace->fbits -= 1;
	}
	return;
}

static int
input_bit(ac_decoder *acd)
{
	int t;

	if (acd->bits_to_go == 0)  {
		acd->buffer = getc(acd->fp);
		if (acd->buffer == EOF)  {
			acd->garbage_bits += 1;
			if (acd->garbage_bits>Code_value_bits - 2)
				error("arithmetic decoder bad input file");
		}
		acd->bits_to_go = 8;
	}

	t = acd->buffer & 1;
	acd->buffer >>= 1;
	acd->bits_to_go -= 1;

	return t;
}

static void
update_model(ac_model *acm, int sym)
{
	int i;

	if (acm->cfreq[0] == Max_frequency)  {
		int cum = 0;
		acm->cfreq[acm->nsym] = 0;
		for (i = acm->nsym - 1; i >= 0; i--)  {
			acm->freq[i] = (acm->freq[i] + 1) / 2;
			cum += acm->freq[i];
			acm->cfreq[i] = cum;
		}
	}

	acm->freq[sym] += 1;
	for (i = sym; i >= 0; i--)
		acm->cfreq[i] += 1;

	return;
}

void
ac_encoder_init(ac_encoder *ace, const char *fn)
{

	if (fn)  {
		ace->fp = fopen(fn, "wb"); /* open in binary mode */
		//fseek(ace->fp, 6, SEEK_SET);
		check(!ace->fp, "arithmetic encoder could not open file");
	}
	else  {
		ace->fp = NULL;
	}

	ace->bits_to_go = 8;

	ace->low = 0;
	ace->high = Top_value;
	ace->fbits = 0;
	ace->buffer = 0;

	ace->total_bits = 0;

	return;
}

void
ac_encoder_done(ac_encoder *ace)
{
	ace->fbits += 1;
	if (ace->low < First_qtr)
		bit_plus_follow(ace, 0);
	else
		bit_plus_follow(ace, 1);
	if (ace->fp)  {
		putc(ace->buffer >> ace->bits_to_go, ace->fp);
		fclose(ace->fp);
	}

	return;
}

void
ac_decoder_init(ac_decoder *acd, const char *fn)
{
	int i;

	acd->fp = fopen(fn, "rb"); /* open in binary mode */
	check(!acd->fp, "arithmetic decoder could not open file");

	acd->bits_to_go = 0;
	acd->garbage_bits = 0;

	acd->value = 0;
	for (i = 1; i <= Code_value_bits; i++)  {
		acd->value = 2 * acd->value + input_bit(acd);
	}
	acd->low = 0;
	acd->high = Top_value;

	return;
}

void
ac_decoder_done(ac_decoder *acd)
{
	fclose(acd->fp);

	return;
}

void
ac_model_init(ac_model *acm, int nsym, int *ifreq, int adapt)
{
	int i;

	acm->nsym = nsym;
	acm->freq = (int *)(void *)calloc(nsym, sizeof(int));
	check(!acm->freq, "arithmetic coder model allocation failure");
	acm->cfreq = (int *)(void *)calloc(nsym + 1, sizeof(int));
	check(!acm->cfreq, "arithmetic coder model allocation failure");
	acm->adapt = adapt;
	//if ifreq(initial frequency) is defined, use the initial frequency
	//else default is that every symbol has the same frequency
	if (ifreq)  {
		acm->cfreq[acm->nsym] = 0;
		for (i = acm->nsym - 1; i >= 0; i--)  {
			acm->freq[i] = ifreq[i];
			acm->cfreq[i] = acm->cfreq[i + 1] + acm->freq[i];
		}
		if (acm->cfreq[0] > Max_frequency)
			error("arithmetic coder model max frequency exceeded");
	}
	else  {
		for (i = 0; i<acm->nsym; i++) {
			acm->freq[i] = 1;
			acm->cfreq[i] = acm->nsym - i;
		}
		acm->cfreq[acm->nsym] = 0;
	}

	return;
}

void
ac_model_done(ac_model *acm)
{
	acm->nsym = 0;
	free(acm->freq);
	acm->freq = NULL;
	free(acm->cfreq);
	acm->cfreq = NULL;

	return;
}

long
ac_encoder_bits(ac_encoder *ace)
{
	return ace->total_bits;
}

void
ac_encode_symbol(ac_encoder *ace, ac_model *acm, int sym)
{
	long range;

	check(sym<0 || sym >= acm->nsym, "symbol out of range");

	range = (long)(ace->high - ace->low) + 1;
	ace->high = ace->low + (range*acm->cfreq[sym]) / acm->cfreq[0] - 1;
	ace->low = ace->low + (range*acm->cfreq[sym + 1]) / acm->cfreq[0];

	for (;;)  {
		if (ace->high<Half)  {
			bit_plus_follow(ace, 0);
		}
		else if (ace->low >= Half)  {
			bit_plus_follow(ace, 1);
			ace->low -= Half;
			ace->high -= Half;
		}
		else if (ace->low >= First_qtr && ace->high<Third_qtr)  {
			ace->fbits += 1;
			ace->low -= First_qtr;
			ace->high -= First_qtr;
		}
		else
			break;
		ace->low = 2 * ace->low;
		ace->high = 2 * ace->high + 1;
	}

	if (acm->adapt)
		update_model(acm, sym);

	return;
}

int
ac_decode_symbol(ac_decoder *acd, ac_model *acm)
{
	long range;
	int cum;
	int sym;

	range = (long)(acd->high - acd->low) + 1;
	cum = (((long)(acd->value - acd->low) + 1)*acm->cfreq[0] - 1) / range;

	for (sym = 0; acm->cfreq[sym + 1]>cum; sym++)
		/* do nothing */;

	check(sym<0 || sym >= acm->nsym, "symbol out of range");

	acd->high = acd->low + (range*acm->cfreq[sym]) / acm->cfreq[0] - 1;
	acd->low = acd->low + (range*acm->cfreq[sym + 1]) / acm->cfreq[0];

	for (;;)  {
		if (acd->high<Half)  {
			/* do nothing */
		}
		else if (acd->low >= Half)  {
			acd->value -= Half;
			acd->low -= Half;
			acd->high -= Half;
		}
		else if (acd->low >= First_qtr && acd->high<Third_qtr)  {
			acd->value -= First_qtr;
			acd->low -= First_qtr;
			acd->high -= First_qtr;
		}
		else
			break;
		acd->low = 2 * acd->low;
		acd->high = 2 * acd->high + 1;
		acd->value = 2 * acd->value + input_bit(acd);
	}

	if (acm->adapt)
		update_model(acm, sym);

	return sym;
}
//******************************算術編碼***************************************

//*****************************Golomb Code*********************************

static void G_output_bit(ag_encoder *, int);
static void G_bit_plus_follow(ag_encoder *, int);
static int G_input_bit(ag_decoder *);
static void G_update_model(ag_encoder *, int);

void AG_encoder_init(ag_encoder *age, const char *fn){

	if (fn)  {
		age->fp = fopen(fn, "wb"); /* open in binary mode */
		check(!age->fp, "arithmetic encoder could not open file");
	}
	else  {
		age->fp = NULL;
	}
	age->buffer = 0;
	age->bits_to_go = 8;
	age->total_bits = 0;
	return;
}
unsigned long  GetGroupID(unsigned long n, unsigned long k){
	//Only for n >0 
	//int k = 2;
	return n >> k;
	/*exp
	while (n > (1 << k) - 2)
	k++;
	return (k - 1); //1,2,3,...
	*/
}
static void UnaryEncode(ag_encoder *age, int n){
	while (n > 0){
		G_output_bit(age, 1);
		n--;
	}
	G_output_bit(age, 0);
}
static void BinaryEncode(ag_encoder *age, int id, int n){
	while (n > 0){
		G_output_bit(age, ((id >> (n - 1)) & 0x01));
		n--;
	}
}
static void G_output_bit(ag_encoder *age, int bit){
	age->buffer <<= 1;
	if (bit)
		age->buffer |= 0x01;//add 1
	age->bits_to_go -= 1;
	age->total_bits += 1;
	if (age->bits_to_go == 0)  {
		if (age->fp)
			putc(age->buffer, age->fp);
		age->bits_to_go = 8;
	}
	return;
}
void AG_encoder_done(ag_encoder *age)
{
	if (age->fp)  {
		putc(age->buffer << age->bits_to_go, age->fp);
		fclose(age->fp);
	}
	return;
}

void AG_decoder_init(ag_decoder *agd, const char *fn)
{
	int i;

	agd->fp = fopen(fn, "rb"); /* open in binary mode */
	check(!agd->fp, "arithmetic decoder could not open file");
	/*
	for (i = 1; i <= Code_value_bits; i++)  {
	agd->value = 2 * agd->value + input_bit(agd);
	}
	*/
	agd->bits_to_go = 0;
	agd->garbage_bits = 0;
	agd->buffer = 0;
	agd->value = 0;

	return;
}
static int G_input_bit(ag_decoder *agd)
{
	int t;

	if (agd->bits_to_go == 0)  {
		agd->buffer = getc(agd->fp);
		if (agd->buffer == EOF)  {
			agd->garbage_bits += 1;
			if (agd->garbage_bits>Code_value_bits - 2)
				error("arithmetic decoder bad input file");
		}
		agd->bits_to_go = 8;
	}

	t = (agd->buffer & 0x80) >> 7;
	agd->buffer <<= 1;
	agd->bits_to_go -= 1;

	return t;
}
int UnaryDecode(ag_decoder *agd){
	int n = 0;
	while (G_input_bit(agd) == 1)
		n++;
	return n;
}
int BinaryDecode(ag_decoder *agd, int n){
	int temp = 0;
	while (n > 0){
		temp |= (G_input_bit(agd) << (n - 1));
		n--;
	}
	return temp;
}

void AG_decoder_done(ag_decoder *agd)
{
	fclose(agd->fp);
	return;
}

void AG_model_init(ag_model *agm,int k, int adapt)
{
	agm->nsym = 0;
	agm->asym = 0;
	agm->adapt = adapt;
	agm->k = 0;// (int)ceilf((float)(logf(k)) / (LN2));
	return;
}

static void update_model(ag_model *agm, int asym,int n)
{
	/*
	if (agm->nsym ==128){
		agm->nsym = floor(agm->nsym / 2);
		agm->asym = floor(agm->asym / 2);
	}
	else{
		agm->asym += sym;
		agm->nsym++;
	}
	*/
	int k = (int)ceilf((float)(logf(asym / (1 * (n))) / (LN2)));
	k > 2 ? agm->k = k : 0;
	//printf("A:%d,N:%d,k:%d\n", asym, n, agm->k);

	return;

}

void AG_model_done(ag_model *agm)
{
	agm->nsym = 0;
	agm->asym = 0;
	return;
}



void GolomEncode(ag_encoder *age, ag_model *agm, unsigned long n){
	unsigned long  GroupID = 0, Index = 0;
	//if (n == 0)
	//	output_bit(age, 0);
	//else{
	GroupID = GetGroupID(n, agm->k);
	Index = n - ((1 << agm->k))*GroupID;
	UnaryEncode(age, GroupID);
	BinaryEncode(age, Index, agm->k);
	//}
	//if (agm->adapt)
	//	update_model(agm, n);
	//printf("GroupID : %ld, Index : %ld,k:%d,a:%d\n", GroupID, Index, agm->k, agm->asym);
}
int  GolomDecode(ag_decoder *agd, ag_model *agm){
	int n2 = 0;
	unsigned long  GroupID2 = 1, Index2 = 1;
	GroupID2 = UnaryDecode(agd);
	//if (GroupID2 != 0){
	Index2 = BinaryDecode(agd, agm->k);
	n2 = ((GroupID2) << agm->k) + Index2;
	//}
	//printf("GroupID2 : %ld, Index : %ld,k:%d,a:%d\n", GroupID2, Index2, agm->k, agm->asym);
	//if (agm->adapt)
	//	update_model(agm, n2);
	return n2;
}
//*****************************Golomb Code*********************************


/****************************************SVD副程式開始***************************************************/
void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
	fprintf(stderr, "Numerical Recipes run-time error...\n");
	fprintf(stderr, "%s\n", error_text);
	fprintf(stderr, "...now exiting to system...\n");
	getchar();
	exit(1);
}

float *vectorsvd(long nl, long nh)
/* allocate a float vector with subscript range v[nl..nh] */
{
	float *v;

	v = (float *)malloc((size_t)((nh - nl + 1 + NR_END)*sizeof(float)));
	if (!v) nrerror("allocation failure in vector()");
	return v - nl + NR_END;
}

void free_vector(float *v, long nl, long nh)
/* free a float vector allocated with vector() */
{
	free((FREE_ARG)(v + nl - NR_END));
}

float pythag(float a, float b)
{
	//computes (a^2+b^2)^1/2 without destructive underflow or overflow
	float absa, absb;
	absa = fabs(a);
	absb = fabs(b);
	if (absa > absb)
		return absa*sqrt(1.0 + SQR(absb / absa));

	else
		return (absb == 0.0 ? 0.0 : absb*sqrt(1.0 + SQR(absa / absb)));
}
void svbksb(float **u, float w[], float **v, int m, int n, float b[], float x[])
{
	int jj, j, i;
	float s, *tmp;
	tmp = vectorsvd(1, n);
	for (j = 1; j <= n; j++)
	{
		s = 0.0;//calculate U transpose multiply B
		if (w[j])
		{      // Nonzero result only if wj is nonzero.  
			for (i = 1; i <= m; i++)
				s += u[i][j] * b[i];
			s /= w[j];        //That is the divide by wj.
		}
		tmp[j] = s;
	}
	for (j = 1; j <= n; j++)
	{                  //Matrix multiply by V to get answer.
		s = 0.0;
		for (jj = 1; jj <= n; jj++)
			s += v[j][jj] * tmp[jj];
		x[j] = s;
	}
	free_vector(tmp, 1, n);
}
void svdcmp(float **a, int m, int n, float w[], float **v)
{
	float pythag(float a, float b);
	int flag, i, its, j, jj, k, l, nm;
	float anorm, c, f, g, h, s, scale, x, y, z, *rv1;

	rv1 = vectorsvd(1, n);
	g = scale = anorm = 0.0;
	for (i = 1; i <= n; i++)
	{
		l = i + 1;
		rv1[i] = scale*g;
		g = s = scale = 0.0;
		if (i <= m) {
			for (k = i; k <= m; k++) scale += fabs(a[k][i]);
			if (scale) {
				for (k = i; k <= m; k++) {
					a[k][i] /= scale;
					s += a[k][i] * a[k][i];
				}
				f = a[i][i];
				g = -SIGN(sqrt(s), f);
				h = f*g - s;
				a[i][i] = f - g;
				for (j = l; j <= n; j++) {
					for (s = 0.0, k = i; k <= m; k++) s += a[k][i] * a[k][j];
					f = s / h;
					for (k = i; k <= m; k++)
						a[k][j] += f*a[k][i];
				}
				for (k = i; k <= m; k++)
					a[k][i] *= scale;
			}
		}
		w[i] = scale *g;
		g = s = scale = 0.0;
		if (i <= m && i != n) {
			for (k = l; k <= n; k++) scale += fabs(a[i][k]);
			if (scale) {

				for (k = l; k <= n; k++) {
					a[i][k] /= scale;
					s += a[i][k] * a[i][k];
				}
				f = a[i][l];
				g = -SIGN(sqrt(s), f);
				h = f*g - s;
				a[i][l] = f - g;
				for (k = l; k <= n; k++)
					rv1[k] = a[i][k] / h;
				for (j = l; j <= m; j++)
				{
					for (s = 0.0, k = l; k <= n; k++)
						s += a[j][k] * a[i][k];
					for (k = l; k <= n; k++)
						a[j][k] += s*rv1[k];
				}
				for (k = l; k <= n; k++) a[i][k] *= scale;
			}
		}
		anorm = FMAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
	}
	for (i = n; i >= 1; i--)
	{    // Accumulation of right-hand transformations.

		if (i < n) {
			if (g) {
				for (j = l; j <= n; j++) //double division to avoid possible underflow.  
					v[j][i] = (a[i][j] / a[i][l]) / g;
				for (j = l; j <= n; j++)
				{
					for (s = 0.0, k = l; k <= n; k++)
						s += a[i][k] * v[k][j];
					for (k = l; k <= n; k++)
						v[k][j] += s*v[k][i];
				}
			}
			for (j = l; j <= n; j++)
				v[i][j] = v[j][i] = 0.0;
		}
		v[i][i] = 1.0;
		g = rv1[i];
		l = i;
	}
	for (i = IMIN(m, n); i >= 1; i--)
	{ //accumulation of left-hand transformations. 	
		l = i + 1;
		g = w[i];
		for (j = l; j <= n; j++)
			a[i][j] = 0.0;
		if (g) {
			g = 1.0 / g;
			for (j = l; j <= n; j++) {
				for (s = 0.0, k = l; k <= m; k++) s += a[k][i] * a[k][j];
				f = (s / a[i][i])*g;
				for (k = i; k <= m; k++) a[k][j] += f*a[k][i];
			}
			for (j = i; j <= m; j++) a[j][i] *= g;
		}
		else for (j = i; j <= m; j++) a[j][i] = 0.0;
		++a[i][i];
	}
	for (k = n; k >= 1; k--)
	{                  //Diagonalization of the bidiagonal form;Loop over
		for (its = 1; its <= 30; its++) {    //singular values, and over alloowed iterations. 	  	

			flag = 1;
			for (l = k; l >= 1; l--) { //Test for splitting.
				nm = l - 1;             //Note that rv1[1] is always zero.  	

				if ((float)(fabs(rv1[l]) + anorm) == anorm) {
					flag = 0;
					break;
				}
				if ((float)(fabs(w[nm]) + anorm) == anorm) break;
			}
			if (flag)
			{
				c = 0.0;                        //Cancellation of rv1[1], if l>1.
				s = 1.0;
				for (i = l; i <= k; i++)
				{
					f = s*rv1[i];
					rv1[i] = c*rv1[i];
					if ((float)(fabs(f) + anorm) == anorm)
						break;
					g = w[i];
					h = pythag(f, g);
					w[i] = h;
					h = 1.0 / h;
					c = g*h;
					s = -f*h;
					for (j = 1; j <= m; j++)
					{
						y = a[j][nm];
						z = a[j][i];
						a[j][nm] = y*c + z*s;
						a[j][i] = z*c - y*s;
					}
				}
			}
			z = w[k];
			if (l == k)
			{                      //convergence.
				if (z < 0.0)
				{                //Singular value is made nonnegtive.
					w[k] = -z;
					for (j = 1; j <= n; j++)
						v[j][k] = -v[j][k];
				}
				break;
			}
			// if (its == 30);
			// nrerror("no convergence in 30 svdcmp iterations");
			x = w[l];                 //shift from bottom 2 by 2 minor 
			nm = k - 1;
			y = w[nm];
			g = rv1[nm];
			h = rv1[k];
			f = ((y - z)*(y + z) + (g - h)*(g + h)) / (2.0*h*y);
			g = pythag(f, 1.0);
			f = ((x - z)*(x + z) + h*((y / (f + SIGN(g, f))) - h)) / x;
			s = 1.0;                                    //next QR transformation
			c = s;
			for (j = l; j <= nm; j++)
			{
				i = j + 1;
				g = rv1[i];
				y = w[i];
				h = s*g;
				g = c*g;
				z = pythag(f, h);
				rv1[j] = z;
				c = f / z;
				s = h / z;
				f = x*c + g*s;
				g = g*c - x*s;
				h = y*s;
				y *= c;
				for (jj = 1; jj <= n; jj++)
				{
					x = v[jj][j];
					z = v[jj][i];
					v[jj][j] = x*c + z*s;
					v[jj][i] = z*c - x*s;
				}
				z = pythag(f, h);
				w[j] = z;            //Rotation can be arbitrary if z=o.
				if (z) {
					z = 1.0 / z;
					c = f*z;
					s = h*z;
				}
				f = c*g + s*y;
				x = c*y - s*g;
				for (jj = 1; jj <= m; jj++)
				{
					y = a[jj][j];
					z = a[jj][i];
					a[jj][j] = y*c + z*s;
					a[jj][i] = z*c - y*s;
				}
			}
			rv1[l] = 0.0;
			rv1[k] = f;
			w[k] = x;
		}
	}
	free_vector(rv1, 1, n);

}
/****************************************SVD副程式開始**************************************************/
int cholesky(float *aa, float *bb){
	int i, j, k;
	int cholesky_nn = predictor_order;
	float sum, p[MAX_nn_size], a[MAX_nn_size][MAX_nn_size], b[MAX_nn_size], x[MAX_nn_size];
	for (i = 0; i<cholesky_nn; i++){
		for (j = 0; j<cholesky_nn; j++){
			a[i][j] = *(aa + i*cholesky_nn + j);
		}
		b[i] = *(bb + i);
	}
	for (i = 0; i <= cholesky_nn - 1; i++)
	{
		for (j = i; j <= cholesky_nn - 1; j++)
		{
			for (sum = a[i][j], k = i - 1; k >= 0; k--)
				sum -= a[i][k] * a[j][k];
			if (i == j)
			{
				if (sum <= 0.0) //a, with rounding errors, is not positive definite.
				{
					//printf("choldc failed\n");
					return -1;
					exit;
				}
				p[i] = sqrt(sum);
			}
			else a[j][i] = sum / p[i];
		}
	}

	for (i = 0; i <= cholesky_nn - 1; i++)
	{
		//Solve L · y = b, storing y in x.
		for (sum = b[i], k = i - 1; k >= 0; k--) sum -= a[i][k] * x[k];
		x[i] = sum / p[i];
	}
	for (i = cholesky_nn - 1; i >= 0; i--)
	{
		//Solve LT · x = y.
		for (sum = x[i], k = i + 1; k <= cholesky_nn - 1; k++)
			sum -= a[k][i] * x[k];
		x[i] = sum / p[i];
	}
	for (i = 0; i <= cholesky_nn - 1; i++){
		a16[i] = x[i]; //updata 
	}
	//printf("choldc sucessful\n");
	return 0;
}
int svd(float *aa, float *bb){
	int i, j, q;
	int N = predictor_order + 1;

	float w[MAX_nn_size], **v, x[MAX_nn_size], wmax, wmin, **a;
	float test[MAX_nn_size][MAX_nn_size] = { { 0 } }, b[MAX_nn_size] = { 0 };


	a = (float**)calloc(N, sizeof(float*));
	for (q = 0; q<N; q++)
		a[q] = (float*)calloc(N, sizeof(float));

	v = (float**)calloc(N, sizeof(float*));
	for (q = 0; q<N; q++)
		v[q] = (float*)calloc(N, sizeof(float));

	for (i = 1; i<N; i++){
		for (j = 1; j<N; j++){
			test[i][j] = *(aa + (i - 1)*(N - 1) + (j - 1));
		}
		b[i] = *(bb + i - 1);
	}
	for (i = 0; i<N; i++)
		for (j = 0; j<N; j++)
			a[i][j] = test[i][j];
	svdcmp(a, N - 1, N - 1, w, v);

	wmax = 0.0;
	for (j = 1; j<N; j++)
		if (w[j]> wmax)
			wmax = w[j];
	wmin = wmax*pow(10.0, -6);
	for (j = 1; j<N; j++)
		if (w[j] < wmin)
			w[j] = 0.0;
	svbksb(a, w, v, N - 1, N - 1, b, x);
	for (i = 1; i<N; i++)
		a16[i - 1] = x[i];

	for (q = 0; q<N; q++)
		free(a[q]);
	free(a);
	for (q = 0; q<N; q++)
		free(v[q]);
	free(v);
	return 0;
}
/****************************************SVD副程式開始**************************************************/

/****************************************TranArea******************************************************/
//建置最小平方法
void normalEquation(float *P, float *y, int Tsize){
	normal_start = clock();//timer
	float *C, *B;
	C = (float *)malloc(sizeof(float)*(predictor_order));								//建構Normal Equation Bx=C
	B = (float *)malloc(sizeof(float)*(predictor_order*predictor_order));				//建構Normal Equation Bx=C
	memset(C, 0.0, sizeof(float)*(predictor_order));
	memset(B, 0.0, sizeof(float)*(predictor_order*predictor_order));
	for (int i = 0; i<predictor_order; i++){
		for (int j = 0; j<predictor_order; j++){
			for (int k = 0; k<Tsize; k++){
				*(B + i * predictor_order + j) += (*(P + k * predictor_order + i))*(*(P + k * predictor_order + j)); //Pt x P
			}
		}
	}
	for (int i = 0; i<predictor_order; i++){
		for (int j = 0; j<Tsize; j++){
			*(C + i) += (*(P + j * predictor_order + i))*(*(y + j));//Pt x y
		}
	}
	normal_end = clock();//timer
	choleskyCount++;
	if (cholesky(B, C) != 0){
		svd(B, C);
		svdCount++;
	}
	free(B);
	free(C);
}
//建構訓練區間
void tranAreaRegular4(int a, int *image){

	TranSize_start = clock();//timer
	int StartX, StartY;
	int EndX, EndY;
	int n = a % (Width);//x
	int m = floor(a / (Width));//y

	if (m <= 7){
		StartY = 2;
		EndY = m;
		if (n <= 7){
			StartX = 2;
			EndX = n + 6;
		}
		else if ((n >= (Width - 7))){
			StartX = n - 6;
			EndX = Width - 2;
		}
		else{
			StartX = n - 6;
			EndX = n + 6;
		}
	}
	else{
		StartY = m - 6;
		EndY = m;
		if (n <= 7){
			StartX = 2;
			EndX = n + 6;
		}
		else if ((n >= (Width - 7))){
			StartX = n - 6;
			EndX = Width - 2;
		}
		else{
			StartX = n - 6;
			EndX = n + 6;
		}
	}


	int TSize = (EndY - StartY) * (EndX - StartX + 1) + (n - StartX); //Traning Sizes
	TranSize_end = clock();//timer

	/*
	printf("*****************************\n");
	printf("x:%d\ty:%d\n",n,m);
	printf("StartX:%d\tEndX:%d\n",StartX,EndX);
	printf("StartY:%d\tEndY:%d\n",StartY,EndY);
	printf("TSize:%d\n",TSize);
	printf("*****************************\n");
	getchar();
	*/
	TranArea_start = clock();//timer
	float *P, *c;
	P = (float *)malloc(sizeof(float)*(TSize*predictor_order));
	c = (float *)malloc(sizeof(float)*(TSize));
	memset(P, 0.0, sizeof(float)*(TSize*predictor_order));
	memset(c, 0.0, sizeof(float)*(TSize));

	int x, y;
	int index = 0;

	for (y = StartY; y <= (EndY - 1); y++){
		for (x = StartX; x <= EndX; x++){
			*(P + index*predictor_order) = *(image + (x + y * Width) - 1);//a 
			*(P + index*predictor_order + 1) = *(image + (x + y * Width) - Width);//b
			*(P + index*predictor_order + 2) = *(image + (x + y * Width) - Width - 1);//c
			*(P + index*predictor_order + 3) = *(image + (x + y * Width) - Width + 1);//d
			*(c + index) = *(image + (x + y * Width));
			index++;
		}
	}
	y = EndY;
	for (x = StartX; x <= (n - 1); x++){
		*(P + index*predictor_order) = *(image + (x + y * Width) - 1);//a
		*(P + index*predictor_order + 1) = *(image + (x + y * Width) - Width);//b
		*(P + index*predictor_order + 2) = *(image + (x + y * Width) - Width - 1);//c
		*(P + index*predictor_order + 3) = *(image + (x + y * Width) - Width + 1);//d
		*(c + index) = *(image + (x + y * Width));
		index++;
	}
	TranArea_end = clock();//timer
	normalEquation(P, c, TSize);
	free(P);
	free(c);
}
void tranAreaRegular6(int a, int *image){
	TranSize_start = clock();//timer
	int StartX, StartY;
	int EndX, EndY;
	int n = a % (Width);//x
	int m = floor(a / (Width));//y

	if (m <= 7){
		StartY = 2;
		EndY = m;
		if (n <= 7){
			StartX = 2;
			EndX = n + 6;
		}
		else if ((n >= (Width - 7))){
			StartX = n - 6;
			EndX = Width - 2;
		}
		else{
			StartX = n - 6;
			EndX = n + 6;
		}
	}
	else{
		StartY = m - 6;
		EndY = m;
		if (n <= 7){
			StartX = 2;
			EndX = n + 6;
		}
		else if ((n >= (Width - 7))){
			StartX = n - 6;
			EndX = Width - 2;
		}
		else{
			StartX = n - 6;
			EndX = n + 6;
		}
	}


	int TSize = (EndY - StartY) * (EndX - StartX + 1) + (n - StartX); //Traning Sizes
	//printf("x:%d\ty:%d\n",n,m);
	//printf("StartX:%d\tEndX:%d\n",StartX,EndX);
	//printf("StartY:%d\tEndY:%d\n",StartY,EndY);
	//printf("TSize:%d\n",TSize);
	TranSize_end = clock();//timer

	TranArea_start = clock();//timer
	float *P, *c;
	P = (float *)malloc(sizeof(float)*(TSize*predictor_order));
	c = (float *)malloc(sizeof(float)*(TSize));
	memset(P, 0.0, sizeof(float)*(TSize*predictor_order));
	memset(c, 0.0, sizeof(float)*(TSize));

	int x, y;
	int index = 0;

	for (y = StartY; y <= (EndY - 1); y++){
		for (x = StartX; x <= EndX; x++){
			*(P + index*predictor_order) = *(image + (x + y * Width) - 1);//a
			*(P + index*predictor_order + 1) = *(image + (x + y * Width) - Width);//b
			*(P + index*predictor_order + 2) = *(image + (x + y * Width) - Width - 1);//c
			*(P + index*predictor_order + 3) = *(image + (x + y * Width) - Width + 1);//d
			*(P + index*predictor_order + 4) = *(image + (x + y * Width) - 2);//e
			*(P + index*predictor_order + 5) = *(image + (x + y * Width) - 2 * Width);//f
			*(c + index) = *(image + (x + y * Width));
			index++;
		}
	}
	y = EndY;
	for (x = StartX; x <= (n - 1); x++){
		*(P + index*predictor_order) = *(image + (x + y * Width) - 1);//a
		*(P + index*predictor_order + 1) = *(image + (x + y * Width) - Width);//b
		*(P + index*predictor_order + 2) = *(image + (x + y * Width) - Width - 1);//c
		*(P + index*predictor_order + 3) = *(image + (x + y * Width) - Width + 1);//d
		*(P + index*predictor_order + 4) = *(image + (x + y * Width) - 2);//e
		*(P + index*predictor_order + 5) = *(image + (x + y * Width) - 2 * Width);//f
		*(c + index) = *(image + (x + y * Width));
		index++;
	}
	TranArea_end = clock();//timer
	normalEquation(P, c, TSize);
	free(P);
	free(c);
}
void tranAreaRegular8(int a, int *image){
	TranSize_start = clock();//timer
	int StartX, StartY;
	int EndX, EndY;
	int n = a % (Width);//x
	int m = floor(a / (Width));//y

	if (m <= 7){
		StartY = 2;
		EndY = m;
		if (n <= 7){
			StartX = 2;
			EndX = n + 6;
		}
		else if ((n >= (Width - 7))){
			StartX = n - 6;
			EndX = Width - 2;
		}
		else{
			StartX = n - 6;
			EndX = n + 6;
		}
	}
	else{
		StartY = m - 6;
		EndY = m;
		if (n <= 7){
			StartX = 2;
			EndX = n + 6;
		}
		else if ((n >= (Width - 7))){
			StartX = n - 6;
			EndX = Width - 2;
		}
		else{
			StartX = n - 6;
			EndX = n + 6;
		}
	}


	int TSize = (EndY - StartY) * (EndX - StartX + 1) + (n - StartX); //Traning Sizes
	//printf("x:%d\ty:%d\n",n,m);
	//printf("StartX:%d\tEndX:%d\n",StartX,EndX);
	//printf("StartY:%d\tEndY:%d\n",StartY,EndY);
	//printf("TSize:%d\n",TSize);
	TranSize_end = clock();//timer

	TranArea_start = clock();//timer
	float *P, *c;
	P = (float *)malloc(sizeof(float)*(TSize*predictor_order));
	c = (float *)malloc(sizeof(float)*(TSize));
	memset(P, 0.0, sizeof(float)*(TSize*predictor_order));
	memset(c, 0.0, sizeof(float)*(TSize));

	int x, y;
	int index = 0;

	for (y = StartY; y <= (EndY - 1); y++){
		for (x = StartX; x <= EndX; x++){
			*(P + index*predictor_order) = *(image + (x + y * Width) - 1);//a
			*(P + index*predictor_order + 1) = *(image + (x + y * Width) - Width);//b
			*(P + index*predictor_order + 2) = *(image + (x + y * Width) - Width - 1);//c
			*(P + index*predictor_order + 3) = *(image + (x + y * Width) - Width + 1);//d
			*(P + index*predictor_order + 4) = *(image + (x + y * Width) - 2);//e
			*(P + index*predictor_order + 5) = *(image + (x + y * Width) - 2 * Width);//f
			*(P + index*predictor_order + 6) = *(image + (x + y * Width) - Width - 2);//nww
			*(P + index*predictor_order + 7) = *(image + (x + y * Width) - 2 * Width - 1);//nnw
			*(c + index) = *(image + (x + y * Width));
			index++;
		}
	}
	y = EndY;
	for (x = StartX; x <= (n - 1); x++){
		*(P + index*predictor_order) = *(image + (x + y * Width) - 1);//a
		*(P + index*predictor_order + 1) = *(image + (x + y * Width) - Width);//b
		*(P + index*predictor_order + 2) = *(image + (x + y * Width) - Width - 1);//c
		*(P + index*predictor_order + 3) = *(image + (x + y * Width) - Width + 1);//d
		*(P + index*predictor_order + 4) = *(image + (x + y * Width) - 2);//e
		*(P + index*predictor_order + 5) = *(image + (x + y * Width) - 2 * Width);//f
		*(P + index*predictor_order + 6) = *(image + (x + y * Width) - Width - 2);//nww
		*(P + index*predictor_order + 7) = *(image + (x + y * Width) - 2 * Width - 1);//nnw
		*(c + index) = *(image + (x + y * Width));
		index++;
	}
	TranArea_end = clock();//timer
	normalEquation(P, c, TSize);
	free(P);
	free(c);
}
void tranAreaRegular10(int a, int *image){
	TranSize_start = clock();//timer
	int StartX, StartY;
	int EndX, EndY;
	int n = a % (Width);//x
	int m = floor(a / (Width));//y

	if (m <= 7){
		StartY = 2;
		EndY = m;
		if (n <= 7){
			StartX = 2;
			EndX = n + 6;
		}
		else if ((n >= (Width - 7))){
			StartX = n - 6;
			EndX = Width - 3;
		}
		else{
			StartX = n - 6;
			EndX = n + 6;
		}
	}
	else{
		StartY = m - 6;
		EndY = m;
		if (n <= 7){
			StartX = 2;
			EndX = n + 6;
		}
		else if ((n >= (Width - 7))){
			StartX = n - 6;
			EndX = Width - 3;
		}
		else{
			StartX = n - 6;
			EndX = n + 6;
		}
	}


	int TSize = (EndY - StartY) * (EndX - StartX + 1) + (n - StartX); //Traning Sizes
	//printf("x:%d\ty:%d\n",n,m);
	//printf("StartX:%d\tEndX:%d\n",StartX,EndX);
	//printf("StartY:%d\tEndY:%d\n",StartY,EndY);
	//printf("TSize:%d\n",TSize);

	TranSize_end = clock();//timer

	TranArea_start = clock();//timer
	float *P, *c;
	P = (float *)malloc(sizeof(float)*(TSize*predictor_order));
	c = (float *)malloc(sizeof(float)*(TSize));
	memset(P, 0.0, sizeof(float)*(TSize*predictor_order));
	memset(c, 0.0, sizeof(float)*(TSize));

	int x, y;
	int index = 0;

	for (y = StartY; y <= (EndY - 1); y++){
		for (x = StartX; x <= EndX; x++){
			*(P + index*predictor_order) = *(image + (x + y * Width) - 1);//a
			*(P + index*predictor_order + 1) = *(image + (x + y * Width) - Width);//b
			*(P + index*predictor_order + 2) = *(image + (x + y * Width) - Width - 1);//c
			*(P + index*predictor_order + 3) = *(image + (x + y * Width) - Width + 1);//d
			*(P + index*predictor_order + 4) = *(image + (x + y * Width) - 2);//e
			*(P + index*predictor_order + 5) = *(image + (x + y * Width) - 2 * Width);//f
			*(P + index*predictor_order + 6) = *(image + (x + y * Width) - Width - 2);//nww
			*(P + index*predictor_order + 7) = *(image + (x + y * Width) - 2 * Width - 1);//nnw
			*(P + index*predictor_order + 8) = *(image + (x + y * Width) - 2 * Width + 1);//nne
			*(P + index*predictor_order + 9) = *(image + (x + y * Width) - Width + 2);//nee
			*(c + index) = *(image + (x + y * Width));
			index++;
		}
	}
	y = EndY;
	for (x = StartX; x <= (n - 1); x++){
		*(P + index*predictor_order) = *(image + (x + y * Width) - 1);//a
		*(P + index*predictor_order + 1) = *(image + (x + y * Width) - Width);//b
		*(P + index*predictor_order + 2) = *(image + (x + y * Width) - Width - 1);//c
		*(P + index*predictor_order + 3) = *(image + (x + y * Width) - Width + 1);//d
		*(P + index*predictor_order + 4) = *(image + (x + y * Width) - 2);//e
		*(P + index*predictor_order + 5) = *(image + (x + y * Width) - 2 * Width);//f
		*(P + index*predictor_order + 6) = *(image + (x + y * Width) - Width - 2);//nww
		*(P + index*predictor_order + 7) = *(image + (x + y * Width) - 2 * Width - 1);//nnw
		*(P + index*predictor_order + 8) = *(image + (x + y * Width) - 2 * Width + 1);//nne
		*(P + index*predictor_order + 9) = *(image + (x + y * Width) - Width + 2);//nee
		*(c + index) = *(image + (x + y * Width));
		index++;
	}
	TranArea_end = clock();//timer
	normalEquation(P, c, TSize);
	free(P);
	free(c);
}

/****************************************TranArea******************************************************/
//插入排序
void Swap(float* array, int x, int y){
	float temp = array[x];
	array[x] = array[y];
	array[y] = temp;
}
void InsertSort(float* array, int size){
	for (int i = 0; i < size; i++){
		for (int j = 1; j < size - i; j++){
			if (array[j] < array[j - 1]){
				Swap(array, j, j - 1);
			}
		}
	}
}

int preValue(int x1, int x2, int x3, int x4, int x5, int x6, int x7, int x8, int x9, int x10){
	if (predictor_order == 4){
		return (int)(((a16[0] * x1)//a
			+ (a16[1] * x2)//b
			+ (a16[2] * x3)//c
			+ (a16[3] * x4))); //   ) / (a16[0] + a16[1] + a16[2] + a16[3] )); //4 d
	}
	if (predictor_order == 6){
		return (int)(((a16[0] * x1)//a
			+ (a16[1] * x2)//b
			+ (a16[2] * x3)//c
			+ (a16[3] * x4) //   ) / (a16[0] + a16[1] + a16[2] + a16[3] )); //4 d
			//N=6
			+ (a16[4] * x5) //e
			+ (a16[5] * x6))); // (a16[0] + a16[1] + a16[2] + a16[3] + a16[4] + a16[5]));//f
	}
	if (predictor_order == 8){
		return (int)(((a16[0] * x1)//a
			+ (a16[1] * x2)//b
			+ (a16[2] * x3)//c
			+ (a16[3] * x4) //   ) / (a16[0] + a16[1] + a16[2] + a16[3] )); //4 d
			//N=6
			+ (a16[4] * x5) //e
			+ (a16[5] * x6) // (a16[0] + a16[1] + a16[2] + a16[3] + a16[4] + a16[5]));//f
			//N=8
			+ (a16[6] * x7) //7
			+ (a16[7] * x8)));
	}
	if (predictor_order == 10){
		return (int)(((a16[0] * x1)//a
			+ (a16[1] * x2)//b
			+ (a16[2] * x3)//c
			+ (a16[3] * x4) //   ) / (a16[0] + a16[1] + a16[2] + a16[3] )); //4 d
			//N=6
			+ (a16[4] * x5) //e
			+ (a16[5] * x6) // (a16[0] + a16[1] + a16[2] + a16[3] + a16[4] + a16[5]));//f
			//N=8
			+ (a16[6] * x7) //7
			+ (a16[7] * x8)
			//N=10
			+ (a16[8] * x9) //7
			+ (a16[9] * x10)));
	}
}
//OpenCV Slover
void cvSVD(int a, int *image){

	int StartX, StartY;
	int EndX, EndY;
	int n = a % (Width)+1;//x
	int m = a / (Width)+1;//y

	if (m <= 7){
		StartY = 2;
		EndY = m;
		if (n <= 7){
			StartX = 2;
			EndX = n + 6;
		}
		else if ((n >= (Width - 7))){
			StartX = n - 6;
			EndX = Width - 2;
		}
		else{
			StartX = n - 6;
			EndX = n + 6;
		}
	}
	else{
		StartY = m - 6;
		EndY = m;
		if (n <= 7){
			StartX = 2;
			EndX = n + 6;
		}
		else if ((n >= (Width - 7))){
			StartX = n - 6;
			EndX = Width - 2;
		}
		else{
			StartX = n - 6;
			EndX = n + 6;
		}
	}


	int TSize = (EndY - StartY) * (EndX - StartX + 1) + (n - StartX); //Traning Sizes
	//printf("x:%d\ty:%d\n",n,m);
	//printf("StartX:%d\tEndX:%d\n",StartX,EndX);
	//printf("StartY:%d\tEndY:%d\n",StartY,EndY);
	//printf("TSize:%d\n",TSize);
	float *P, *c;
	P = (float *)malloc(sizeof(float)*(TSize*predictor_order));
	c = (float *)malloc(sizeof(float)*(TSize));
	memset(P, 0.0, sizeof(float)*(TSize*predictor_order));
	memset(c, 0.0, sizeof(float)*(TSize));

	int x, y;
	int index = 0;

	for (y = StartY; y <= (EndY - 1); y++){
		for (x = StartX; x <= EndX; x++){
			*(P + index*predictor_order) = *(image + (x + y * Width) - 1);//a
			*(P + index*predictor_order + 1) = *(image + (x + y * Width) - Width);//b
			*(P + index*predictor_order + 2) = *(image + (x + y * Width) - Width - 1);//c
			*(P + index*predictor_order + 3) = *(image + (x + y * Width) - Width + 1);//d
			*(P + index*predictor_order + 4) = *(image + (x + y * Width) - 2);//e
			*(P + index*predictor_order + 5) = *(image + (x + y * Width) - 2 * Width);//f
			*(c + index) = *(image + (x + y * Width));
			index++;
		}
	}
	y = EndY;
	for (x = StartX; x <= (n - 1); x++){
		*(P + index*predictor_order) = *(image + (x + y * Width) - 1);//a
		*(P + index*predictor_order + 1) = *(image + (x + y * Width) - Width);//b
		*(P + index*predictor_order + 2) = *(image + (x + y * Width) - Width - 1);//c
		*(P + index*predictor_order + 3) = *(image + (x + y * Width) - Width + 1);//d
		*(P + index*predictor_order + 4) = *(image + (x + y * Width) - 2);//e
		*(P + index*predictor_order + 5) = *(image + (x + y * Width) - 2 * Width);//f
		*(c + index) = *(image + (x + y * Width));
		index++;
	}

	CvMat *Matrix1 = cvCreateMat(TSize, predictor_order, CV_32FC1);
	CvMat *Matrix2 = cvCreateMat(TSize, 1, CV_32FC1);
	CvMat *SolveSet = cvCreateMat(predictor_order, 1, CV_32FC1);


	//printf("*******************************************************************************\n");

	cvSetData(Matrix1, P, Matrix1->step);
	cvSetData(Matrix2, c, Matrix2->step);
	cvSolve(Matrix1, Matrix2, SolveSet, CV_SVD);

	//PrintMatrix(Matrix1,Matrix1->rows,Matrix1->cols);
	//PrintMatrix(Matrix2,Matrix2->rows,Matrix2->cols);
	//PrintMatrix(SolveSet,SolveSet->rows,SolveSet->cols);
	//printf("%d",Matrix1->rows);
	for (int l = 0; l<predictor_order; l++){
		a16[l] = cvGet1D(SolveSet, l).val[0];;
	}
	//printf("a16: %f,%f,%f,%f,%f,%f\n",a16[0],a16[1],a16[2],a16[3],a16[4],a16[5]);
	free(P);
	free(c);
}

int EdgeDetect(int i, int *image){
	float average, variance, std_deviation, sum = 0, sum1 = 0;
	int x[4];
	int kh[2], kl[2];
	float aveh, varh;
	float avel, varl;
	x[0] = *(image + i - 1);//x1
	x[1] = *(image + i - Width);//x2
	x[2] = *(image + i - Width - 1);//x3
	x[3] = *(image + i - Width + 1);//x4
	average = (x[0] + x[1] + x[2] + x[3]) / (float)4;//x4
	variance = (pow((x[0] - average), 2)//x1
		+ pow((x[1] - average), 2)//x2
		+ pow((x[2] - average), 2)//x3
		+ pow((x[3] - average), 2)) / (float)4;//x4
	for (i = 0; i<4; i++){
		int h = 0, l = 0;
		if (x[i] - average < 0){
			kh[h++];
		}
		if (x[i] - average > 0){
			kl[l++];
		}
	}
	if (variance >= 100){
		aveh = (kh[0] + kh[1]) / (float)2;
		varh = (pow((kh[0] - aveh), 2)
			+ pow((kh[1] - aveh), 2)) / (float)2;

		avel = (kl[0] + kl[1]) / (float)2;
		varl = (pow((kl[0] - avel), 2)
			+ pow((kl[1] - avel), 2)) / (float)2;
		if (variance >= (10 * (varh + varl))){
			return 1;
		}
	}
	return 0;
}
//Medium Edge Dector
int MED(int A, int B, int C){
	if (C >= MAX(A, B)){
		return MIN(A, B);
	}
	else if (C <= MIN(A, B)){
		return MAX(A, B);
	}
	else{
		return A + (B - C);
	}
}

int PredictRange(int xp){
	if (xp > 255)
		return xp = 255;
	else if (xp < 0)
		return xp = 0;
	else
		return xp;
}

unsigned char ErrorStrength(double delta){
	//8 Section
	//*
	if (delta < q[3]){
		if (delta < q[1])
			return (delta < q[0]) ? 0 : 1;
		else
			return (delta < q[2]) ? 2 : 3;
	}
	else{
		if (delta < q[5])
			return (delta < q[4]) ? 4 : 5;
		else
			return (delta < q[6]) ? 6 : 7;
	}
	//*/
	//6 Section
	/*
	if (delta < q[3]){
	if (delta < q[0])
	return 0;
	else if(delta > q[1])
	return 2;
	else
	return 1;
	}
	else{
	if (delta < q[3])
	return 3;
	else if (delta > q[4])
	return 5;
	else
	return 4;
	}
	*/
	//4 Section
	/*
	if (delta < q[1])
	return (delta < q[0]) ? 0 : 1;
	else
	return (delta < q[2]) ? 2 : 3;
	//*/
}

int context(int x, int n, int w, int nw, int ne, int nn, int ww) {
	unsigned char quantizerTexture = 0;
	unsigned char contextIndex = 0;
	//使用過往經驗
	if (n < predicted)		quantizerTexture |= 1;//In			//Context量化
	if (w < predicted)		quantizerTexture |= 2;//Iw
	if (nw < predicted)	quantizerTexture |= 4;//Inw
	if (ne < predicted)	quantizerTexture |= 8;//Ine
	if (nn < predicted)	quantizerTexture |= 16;//Inn
	if (ww >= predicted)			quantizerTexture |= 32;//Iww
	if ((2 * n - nn) < predicted)	quantizerTexture |= 64;//2In-Inn
	if ((2 * w - ww) < predicted)		quantizerTexture |= 128;//2Iw-Iww

	// Compound Context (其中 quantizerEnergy ÷ 2 為 Error Context)
	//contextIndex = (quantizerTexture << 2) + (quantizerEnergy >> 1);
	Se[contextIndex];


	//累計過往經驗
	Se[contextIndex] += (x - predicted);
	context_count[contextIndex] ++;	//Context計算

	if (context_count[contextIndex] == 128) {
		context_count[contextIndex] >>= 1;
		Se[contextIndex] >>= 1;
		return (Se[contextIndex] / context_count[contextIndex]);		//誤差補償
	}
	else {
		if (context_count[contextIndex] != 0){
			return (Se[contextIndex] / context_count[contextIndex]);		//誤差補償
		}
		//context_count[contextIndex] = context_count[contextIndex] + 1;	//Context計算
		//S[contextIndex] += (x - predicted);
		//printf("%d,%d\n",S[Q_estimator][C],context_count[Q_estimator][C]);
		//*(error_image + i) = (*(image + i) - predicted);// +Pixel_Shift;
		//printf("%d\n",S[C]/context_count[C]);
	}
}

void writeArithmeticCoding(int quantizerEnergy, int symbol){
	switch (quantizerEnergy){
	case 0:
		ac_encode_symbol(&ace1, &acm1, symbol);
		Count1++;
		break;
	case 1:
		ac_encode_symbol(&ace1, &acm2, symbol);
		Count2++;
		break;
	case 2:
		ac_encode_symbol(&ace1, &acm3, symbol);
		Count3++;
		break;
	case 3:
		ac_encode_symbol(&ace1, &acm4, symbol);
		Count4++;
		break;
	case 4:
		ac_encode_symbol(&ace1, &acm5, symbol);
		Count5++;
		break;
	case 5:
		ac_encode_symbol(&ace1, &acm6, symbol);
		Count6++;
		break;
	case 6:
		ac_encode_symbol(&ace1, &acm7, symbol);
		Count7++;
		break;
	case 7:
		ac_encode_symbol(&ace1, &acm8, symbol);
		Count8++;
		break;
	case 8:
		ac_encode_symbol(&ace1, &acm9, symbol);
		Count9++;
		break;
	default:
		break;
	}
}
void writeGolombCoding(int quantizerEnergy, int symbol){
	switch (quantizerEnergy){
	case 0:
		GolomEncode(&age1, &agm1, symbol);
		Count1++;
		break;
	case 1:
		GolomEncode(&age1, &agm2, symbol);
		Count2++;
		break;
	case 2:
		GolomEncode(&age1, &agm3, symbol);
		Count3++;
		break;
	case 3:
		GolomEncode(&age1, &agm4, symbol);
		Count4++;
		break;
	case 4:
		GolomEncode(&age1, &agm5, symbol);
		Count5++;
		break;
	case 5:
		GolomEncode(&age1, &agm6, symbol);
		Count6++;
		break;
	case 6:
		GolomEncode(&age1, &agm7, symbol);
		Count7++;
		break;
	case 7:
		GolomEncode(&age1, &agm8, symbol);
		Count8++;
		break;
	case 8:
		GolomEncode(&age1, &agm9, symbol);
		Count9++;
		break;
	default:
		break;
	}
}

void AdatpEncoder(int EncodeValue, int quantizerEnergy){
//quantizerEnergy = 7;
	int escape = N[quantizerEnergy] - 1;

	if (quantizerEnergy <7){
		if (EncodeValue >= escape){
			writeArithmeticCoding(quantizerEnergy, escape);
			AdatpEncoder(EncodeValue - escape, quantizerEnergy++);
		}
		else
			writeArithmeticCoding(quantizerEnergy, EncodeValue);
	}
	else
		writeArithmeticCoding(quantizerEnergy, EncodeValue);
}

void AdatpGEncoder(int EncodeValue, int quantizerEnergy){
	quantizerEnergy = 7;
	int escape = N[quantizerEnergy] - 1;
	//*
	if (quantizerEnergy <7){
		if (EncodeValue >= escape){
			writeGolombCoding(quantizerEnergy, escape);
			AdatpGEncoder(EncodeValue - escape, quantizerEnergy++);
		}
		else
			writeGolombCoding(quantizerEnergy, EncodeValue);
	}
	else
		//*/
		writeGolombCoding(quantizerEnergy, EncodeValue);
}

int quantize(double a)
{
	int NC = 8;
	int i;

	if (a<q[0]) return(0);
	else if (a >= q[NC - 1]) return(NC);
	else
		for (i = 1; i<NC; i++)
			if (a >= q[i - 1] && a<q[i])
				return(i);
}

void RegularMode(int* SourceImage, int* ErrorImage, int* DeltaImage, unsigned char* EdgeImage, char *e){
	printf("Regular Mode....\n");
	int th = 3;
	int error = 0;
	int EncodeValue = 0;
	float a16p[12] = { 0 };


	for (int i = 1; i < (Width)*(Height)*(Pixel_Byte); i++){
		if (predictor_order != PredictorOrderSetting && SettingN){
			printf("Only DO N = %d\n", PredictorOrderSetting);
			break;//do only N
		}

		bool isLSR = false;
		bool isMED = false;
		x = *(SourceImage + i);

		if (((((i % (Width)) != 0)) && ((i >= (Width)))) && !pureMED){	//第0列以及第0行以外
			w = *(SourceImage + i - 1);//1
			n = *(SourceImage + i - Width);//2
			nw = *(SourceImage + i - Width - 1);//3
			ne = *(SourceImage + i - Width + 1);//4
			if (predictor_order == 4){
				if (((i % (Width)) < 2) || (i < (2 * Width)) || ((i % (Width)) >(Width - 2))){//第1列,第1行,Width-1行   MED
					//|| (abs(nw - w) < th && abs(nw - n) < th && abs(n - ne) < th)){
					//if (((i % (Width)) == 1) || (i <= (2 * Width)) || ((i % (Width)) >(Width - 2))){//第1列,第1行,Width-1行 MED

					predicted = MED(w, n, nw);
					isMED = true;
				}
				else{
					if (AllLSR || (abs(n - w)) >= 12//){//至聖方法
						|| abs(*(ErrorImage + i - 1)) >= 8){
						tranAreaRegular4(i, SourceImage);
						TranArea_time += (double)(TranArea_end - TranArea_start) / CLOCKS_PER_SEC;
						TranSize_time += (double)(TranSize_end - TranSize_start) / CLOCKS_PER_SEC;
						normal_time += (double)(normal_end - normal_start) / CLOCKS_PER_SEC;
					}
					predicted = preValue(w, n, nw, ne, 0, 0, 0, 0, 0, 0);
					isLSR = true;
				}
			}
			if (predictor_order == 6){
				if (((i % (Width)) < 2) || (i < (2 * Width)) || ((i % (Width)) >= (Width -1 ))){//第1列,第1行,第2列,第2行,Width-1行   MED

					//|| (abs(nw - w) < th && abs(nw - n) < th && abs(n - ne) < th)){
					//if (((i % (Width)) == 1) || (i <= (2 * Width)) || ((i % (Width)) >(Width - 2))){//第1列,第1行,Width-1行 MED

					predicted = MED(w, n, nw);
					isMED = true;
				}
				/*
				else if (((i % (Width)) < 3) || (i < (3 * Width)) || ((i % (Width)) >= (Width - 2))  && !(((i % (Width)) < 2) || (i < (2 * Width)) || ((i % (Width)) >= (Width - 1)))){
							tranAreaRegular4(i, SourceImage);
							TranArea_time += (double)(TranArea_end - TranArea_start) / CLOCKS_PER_SEC;
							TranSize_time += (double)(TranSize_end - TranSize_start) / CLOCKS_PER_SEC;
							normal_time += (double)(normal_end - normal_start) / CLOCKS_PER_SEC;
						predicted = preValue(w, n, nw, ne, 0, 0, 0, 0, 0, 0);
						isLSR = true;
				}
				*/
				else{
					ww = *(SourceImage + i - 2);//5
					nn = *(SourceImage + i - 2 * Width);//6

					if (AllLSR //|| (abs(n - w)) >= 12//){//至聖方法
						|| abs(*(ErrorImage + i - 1)) >= 10){
						tranAreaRegular6(i, SourceImage);
						TranArea_time += (double)(TranArea_end - TranArea_start) / CLOCKS_PER_SEC;
						TranSize_time += (double)(TranSize_end - TranSize_start) / CLOCKS_PER_SEC;
						normal_time += (double)(normal_end - normal_start) / CLOCKS_PER_SEC;

					}
					predicted = preValue(w, n, nw, ne, ww, nn, 0, 0, 0, 0);
					isLSR = true;

					/*
					if ((i < (8 * Width))){
					tranAreaRegular6(i, SourceImage);
					predicted = preValue(w, n, nw, ne, ww, nn, 0, 0, 0, 0);
					}
					else{
					if ((i % (Width)) < 8){
					tranAreaRegular6(i, SourceImage);
					predicted = preValue(w, n, nw, ne, ww, nn, 0, 0, 0, 0);
					}
					}
					*/
				}
			}
			if (predictor_order == 8){
				if (((i % (Width)) <= 2) || (i < (3 * Width)) || ((i % (Width)) >(Width - 2))){//第1列,第1行,第2列,第2行,Width-1行   MED
					//|| (abs(nw - w) < th && abs(nw - n) < th && abs(n - ne) < th)){
					//if (((i % (Width)) == 1) || (i <= (2 * Width)) || ((i % (Width)) >(Width - 2))){//第1列,第1行,Width-1行 MED

					predicted = MED(w, n, nw);
					isMED = true;
				}
				else{
					ww = *(SourceImage + i - 2);//5
					nn = *(SourceImage + i - 2 * Width);//6
					int nww = *(SourceImage + i - Width - 2);//7
					int nnw = *(SourceImage + i - 2 * Width - 1);//8
					if (AllLSR || (abs(n - w)) >= 12//){//至聖方法
						|| abs(*(ErrorImage + i - 1)) >= 8){
						tranAreaRegular8(i, SourceImage);
						TranArea_time += (double)(TranArea_end - TranArea_start) / CLOCKS_PER_SEC;
						TranSize_time += (double)(TranSize_end - TranSize_start) / CLOCKS_PER_SEC;
						normal_time += (double)(normal_end - normal_start) / CLOCKS_PER_SEC;
					}
					predicted = preValue(w, n, nw, ne, ww, nn, nww, nnw, 0, 0);
					isLSR = true;
				}
			}
			if (predictor_order == 10){
				if (((i % (Width)) <= 2) || (i < (3 * Width)) || ((i % (Width)) >(Width - 3))){//第1列,第1行,第2列,第2行,Width-1行   MED
					//|| (abs(nw - w) < th && abs(nw - n) < th && abs(n - ne) < th)){
					//if (((i % (Width)) == 1) || (i <= (2 * Width)) || ((i % (Width)) >(Width - 2))){//第1列,第1行,Width-1行 MED

					predicted = MED(w, n, nw);
					isMED = true;
				}
				else{
					ww = *(SourceImage + i - 2);//5
					nn = *(SourceImage + i - 2 * Width);//6
					int nww = *(SourceImage + i - Width - 2);//7
					int nnw = *(SourceImage + i - 2 * Width - 1);//8
					int nne = *(SourceImage + i - 2 * Width + 1);//9
					int nee = *(SourceImage + i - Width + 2);//10
					if (AllLSR || (abs(n - w)) >= 12//){//至聖方法
						|| abs(*(ErrorImage + i - 1)) >= 8){
						tranAreaRegular10(i, SourceImage);
						TranArea_time += (double)(TranArea_end - TranArea_start) / CLOCKS_PER_SEC;
						TranSize_time += (double)(TranSize_end - TranSize_start) / CLOCKS_PER_SEC;
						normal_time += (double)(normal_end - normal_start) / CLOCKS_PER_SEC;
					}
					predicted = preValue(w, n, nw, ne, ww, nn, nww, nnw, nne, nee);
					isLSR = true;
				}
			}
		}
		else {
			if (i == 0){//0,0
				predicted = 0;
			}
			else if (i%Width == 0){//0,n 第0行
				predicted = *(SourceImage + i - Width);
			}
			else if (i < Width){//n,0 第0列
				predicted = *(SourceImage + i - 1);
			}
			else if (pureMED){
				w = *(SourceImage + i - 1);//1
				n = *(SourceImage + i - Width);//2
				nw = *(SourceImage + i - Width - 1);//3
				predicted = MED(w, n, nw);
				isMED = true;
			}
		}
		if (predicted<0 || predicted>255)
		{
			predicted = 0;
		   predicted = (n+w+nw+ne+nn+ww)/6;
		}
		predicted = PredictRange(predicted);
		*(ErrorImage + i) = x - (predicted);

		unsigned short quantizerTexture = 0;
		unsigned short contextIndex = 0;
		unsigned char quantizerEnergy = 2;
		double delta = 0;
		int ep = 0;
		int ap = 0;
		float RoundTemp = 0;
		int d = 0;
		int En, Ew, Enw, Ene;
		int level, dist, refDist = 1000;

		if (contextLSR && i > (2 * Width) && (i % Width) > 1 && (i % Width) < (Width - 1)){
		//if (((i % (Width)) > 2) && (i > (3 * Width)) && ((i % (Width)) < (Width - 2))){
			
			//tranAreaRegular6(i, ErrorImage);

			int En = *(ErrorImage + i - Width);
			int Enn = *(ErrorImage + i - 2 * Width);
			int Enw = *(ErrorImage + i - Width - 1);
			int Ew = *(ErrorImage + i - 1);
			int Eww = *(ErrorImage + i - 2);
			int Ene = *(ErrorImage + i - Width + 1);
			int Enne = *(ErrorImage + i - 2 * Width + 1);
			int Ewwn = *(ErrorImage + i - 2 - Width);
			//delta = preValue(Ew, En, Enw, Ene, Eww, Enn, 0, 0, 0, 0);//EDP
			//delta = PredictRange(delta);
			double powd,mean,var;
			powd = pow(abs(En), 2) + pow(abs(Ew), 2) + pow(abs(Enw), 2) + pow(abs(Ene), 2) + pow(abs(Enn), 2) + pow(abs(Eww), 2);// +pow(abs(Enne), 2) + pow(abs(Ewwn), 2);
			//mean = (n+w+nw+ne)/4;
			//var = pow((n - mean), 2) + pow((w - mean), 2) + pow((nw - mean), 2) + pow((ne- mean), 2);
			delta = pow((powd / 6), 0.5);
			//delta = 0.0034 * (abs(ne - n) + abs(n - nw) + abs(nw - w)) + 0.048 * abs(Ew);
			//delta = MED(abs(Ew), abs(En), abs(Enw));
			/*
			if (quantize(delta * 10)<15)
				quantizerEnergy = 0;
			else if (quantize(delta * 10)<50)
				quantizerEnergy = 1;
			*/
			quantizerEnergy = quantize(delta*10);

			if (n < predicted)		quantizerTexture |= 1;//In			//Context量化
			if (w < predicted)		quantizerTexture |= 2;//Iw
			if (nw < predicted)	quantizerTexture |= 4;//Inw
			if (ne < predicted)	quantizerTexture |= 8;//Ine
			if (nn < predicted)	quantizerTexture |= 16;//Inn
			if (ww >= predicted)			quantizerTexture |= 32;//Iww
			if ((2 * n - nn) < predicted)	quantizerTexture |= 64;//2In-Inn
			if ((2 * w - ww) < predicted)		quantizerTexture |= 128;//2Iw-Iww

			// Compound Context (其中 quantizerEnergy ÷ 2 為 Error Context)
			contextIndex = (quantizerTexture << 2) + (quantizerEnergy >> 1);
			//printf("i:%d,quantizerEnergy:%d\n",i,quantizerEnergy);
			if (context_count[contextIndex] > 0)
				ep = Se[contextIndex] / context_count[contextIndex];
		}

		EncodeValue = x - (predicted);
		//*
		int  coperror = predicted + ep;
		coperror = PredictRange(coperror);
		EncodeValue = x - (coperror);

		if (ep < 0)
			EncodeValue = -EncodeValue;
		else
			EncodeValue = EncodeValue;
			//*/
		//*
		if (EncodeValue < -128)
			EncodeValue += 256;
		if (EncodeValue >= 128)
			EncodeValue -= 256;
		//*/

		EncodeValue = (EncodeValue >= 0) ? (EncodeValue << 1) : -(EncodeValue << 1) - 1;

		//EncodeValue -= 128;
		//*/
		//if (EncodeValue < 0)
		//	printf("%d\n", EncodeValue);
		*(DeltaImage + i) = EncodeValue;

		//*
		if (contextLSR && i > (2 * Width) && (i % Width) > 1 && (i % Width) < (Width - 1)){

			Se[contextIndex] += (x - predicted);
			//Sa[contextIndex] += (x - coperror);
			context_count[contextIndex] ++;	//Context計算
			if (context_count[contextIndex] == 128) {
				context_count[contextIndex] >>= 1;
				Se[contextIndex] >>= 1;
				//Sa[contextIndex] >>= 1;
			}
			if (agm1.adapt)
				update_model(&agm1, abs(Se[contextIndex]), context_count[contextIndex]);
		}
		else{
			Se[ContextArraySize-1] += (x - predicted);
			context_count[ContextArraySize - 1] ++;	//Context計算
			if (context_count[ContextArraySize - 1] == 128) {
				context_count[ContextArraySize - 1] >>= 1;
				Se[ContextArraySize - 1] >>= 1;
			}
			if (agm1.adapt)
				update_model(&agm1, abs(Se[ContextArraySize - 1]), context_count[ContextArraySize - 1]);
		}
		//*/




#if ADAPTIVE_ARITHMETIC_CODING
		//quantizerEnergy = 0;
		//if (quantizerEnergy == 5)

		AdatpEncoder(EncodeValue, quantizerEnergy);

		//writeArithmeticCoding(quantizerEnergy, EncodeValue);

#endif
#if ADAPTIVE_GOLOMB_CODING
		//quantizerEnergy = 0;
		//if (quantizerEnergy == 5)

		//AdatpGEncoder(EncodeValue, quantizerEnergy);

		writeGolombCoding(0, EncodeValue);

#endif

		

	}
}

void StandardDeviation(){
	float temp = 0;
	//Average
	for (int i = 0; i <ContextArraySize; i++){
		temp = 0;
		for (int j = times_count[i]; j>0; j--){
			temp += context_val[i][j];
		}
		if (times_count[i] > 0)
			context_avg[i] = temp / times_count[i];
	}
	//Variance
	for (int i = 0; i <ContextArraySize; i++){
		temp = 0;
		for (int j = times_count[i]; j >0; j--){
			temp += pow((context_val[i][j] - context_avg[i]), 2);
		}
		if (times_count[i] > 0)
			context_var[i] = sqrt(temp / times_count[i]);
	}
	//for (int i = 0; i < 100; i++){
	//printf("Context:%d, Times:%d , Avg:%.2f, Var: %.2f \n", i, times_count[i], context_avg[i], context_var[i]);
	//}
	//system("pause");
}

int main(int argc, char *argv[])
{
#if TEST_ONE
	argc = 7;
	argv[1] = "Lennagrey";
	argv[2] = "512";
	argv[3] = "512";
	argv[4] = "6";
#endif

	if (argc == 7) {
		printf("-------------------ENCODING START---------------------------\n");
		FILEName = argv[1];
		Width = strtol(argv[2], NULL, 10);
		Height = strtol(argv[3], NULL, 10);
		predictor_order = strtol(argv[4], NULL, 10);
		printf("%s\t", FILEName);
		printf("%d\n", predictor_order);
		char* InImageName = FileName(FILE_PATH, FILEName, ".bmp");//輸入影像檔名
		char* OutImageName = OutFileName(FILE_OUT_PATH, FILEName, argv[4], "Error.bmp");//輸出影像檔名

		Mat image, gray_image;

		image = imread(InImageName, 1);
		cvtColor(image, gray_image, CV_RGB2GRAY);

		int *SourceImage, *ErrorImage;
		int *DeltaImage;
		unsigned char *EdgeImage;
		char *predictdata, *timetext, *Counttext, *e, *entropytext;

		SourceImage = (int *)malloc(sizeof(int)*((Width)*(Height)*(Pixel_Byte)));  //用以儲存原始影像的空間
		ErrorImage = (int *)malloc(sizeof(int)*((Width)*(Height)*(Pixel_Byte)));  //用以儲存誤差影像的空間
		memset(ErrorImage, 0, sizeof(int)*((Width)*(Height)*(Pixel_Byte)));
		EdgeImage = (unsigned char *)malloc(sizeof(unsigned char)*((Width)*(Height)*(Pixel_Byte)));  //用以儲存觸發邊界影像的空間 
		memset(EdgeImage, 255, sizeof(unsigned char)*((Width)*(Height)*(Pixel_Byte)));
		DeltaImage = (int *)malloc(sizeof(int)*((Width)*(Height)*(Pixel_Byte)));  //用以儲存誤差影像的空間
		memset(DeltaImage, 0, sizeof(int)*((Width)*(Height)*(Pixel_Byte)));

		predictdata = (char *)malloc(sizeof(char)*((Width)*(Height)*(Pixel_Byte)* 8)); //Error Data to txt
		memset(predictdata, 0.0, sizeof(char)*((Width)*(Height)*(Pixel_Byte)* 8));

		timetext = (char *)malloc(sizeof(char) * 30); //time Text out to txt  
		memset(timetext, 0.0, sizeof(char));
		Counttext = (char *)malloc(sizeof(char) * 25);  //count Text out to txt 
		memset(Counttext, 0.0, sizeof(char));
		entropytext = (char *)malloc(sizeof(char) * 10);  //Entropy Text out to txt 
		memset(entropytext, 0.0, sizeof(char));

		e = (char *)malloc(sizeof(char)*((Width)*(Height)*(Pixel_Byte)* 8));  //在CPU配置一塊用以儲存原始影像的空間 
		memset(e, 0.0, sizeof(char)*((Width)*(Height)*(Pixel_Byte)* 8));

		compress_start = clock();	  //抓取程式開始執行之系統時間 
		ReadImage(gray_image.data, SourceImage);
		//*(ErrorImage) = *(SourceImage);
		
#if ADAPTIVE_ARITHMETIC_CODING
		char *c;
		c = (char *)malloc(sizeof(char) * 2);
		sprintf(c, "%d", predictor_order);

		FILE *fp;
		if ((fp = fopen(OutFileName(FILE_OUT_PATH, FILEName, c, "Encode"), "wb")) == 0){
			printf("Write File Header Fail\n");
			system("pause");
		}
		else{
			//fseek(fp, 0, SEEK_SET);
			putc((Width >> 8), fp);
			putc((Width & 0x00FF), fp);
			putc((Height >> 8), fp);
			putc((Height & 0x00FF), fp);
			putc((predictor_order), fp);
			putc(*(SourceImage), fp);

			fclose(fp);
			printf("Write File Header...OK!\n");
		}
		printf("%d\n", *(SourceImage));
		ac_encoder_init(&ace1, OutFileName(FILE_OUT_PATH, FILEName, c, "Encode"));//算數編碼Init
		ac_model_init(&acm1, N[0], NULL, ADAPT);
		ac_model_init(&acm2, N[1], NULL, ADAPT);
		ac_model_init(&acm3, N[2], NULL, ADAPT);
		ac_model_init(&acm4, N[3], NULL, ADAPT);
		
		ac_model_init(&acm5, N[4], NULL, ADAPT);
		ac_model_init(&acm6, N[5], NULL, ADAPT);
		//*
		ac_model_init(&acm7, N[6], NULL, ADAPT);
		ac_model_init(&acm8, N[7], NULL, ADAPT);
		ac_model_init(&acm9, N[7], NULL, ADAPT);
		//*/

		/*
		ac_encode_symbol(&ace1, &acm4, (Width >> 8));//Width Upper 2 Bit
		ac_encode_symbol(&ace1, &acm4, (Width & 0x00FF));//Width Lower 2 Bit
		ac_encode_symbol(&ace1, &acm4, (Height >> 8));//Height Upper 2 Bit
		ac_encode_symbol(&ace1, &acm4, (Height & 0x00FF));//Height Lower 2 Bit
		ac_encode_symbol(&ace1, &acm4, predictor_order);//Predict Size
		*/
#endif

#if ADAPTIVE_GOLOMB_CODING
		char *c;
		c = (char *)malloc(sizeof(char) * 2);
		sprintf(c, "%d", predictor_order);

		FILE *fp;
		if ((fp = fopen(OutFileName(FILE_OUT_PATH, FILEName, c, "Encode"), "wb")) == 0){
			printf("Write File Header Fail\n");
			system("pause");
		}
		else{
			//fseek(fp, 0, SEEK_SET);
			putc((Width >> 8), fp);
			putc((Width & 0x00FF), fp);
			putc((Height >> 8), fp);
			putc((Height & 0x00FF), fp);
			putc((predictor_order), fp);
			putc(*(SourceImage), fp);

			fclose(fp);
			printf("Write File Header...OK!\n");
		}
		printf("%d\n", *(SourceImage));
		AG_encoder_init(&age1, OutFileName(FILE_OUT_PATH, FILEName, c, "Encode"));//算數編碼Init
		AG_model_init(&agm1, 2, 1);

		//*/

		/*
		ac_encode_symbol(&ace1, &acm4, (Width >> 8));//Width Upper 2 Bit
		ac_encode_symbol(&ace1, &acm4, (Width & 0x00FF));//Width Lower 2 Bit
		ac_encode_symbol(&ace1, &acm4, (Height >> 8));//Height Upper 2 Bit
		ac_encode_symbol(&ace1, &acm4, (Height & 0x00FF));//Height Lower 2 Bit
		ac_encode_symbol(&ace1, &acm4, predictor_order);//Predict Size
		*/
#endif

		compress_start2 = clock();
		RegularMode(SourceImage, ErrorImage, DeltaImage, EdgeImage, e);
		compress_end = clock();

#if OUTPUT_BITRATE
		char buffer[10];
		//int size = sprintf(buffer, "%d\n", ac_encoder_bits(&ace1));
		//int size = sprintf(buffer, "%.3f\n", (double)(ac_encoder_bits(&ace1) + 40) / (Width  * Height));
#if ADAPTIVE_ARITHMETIC_CODING
		int size = sprintf(buffer, "%.3f\n", (double)(ac_encoder_bits(&ace1) + 40) / (Width  * Height));
		writeBitRate(buffer, FILEName, size);
#endif
#if ADAPTIVE_GOLOMB_CODING
		int size = sprintf(buffer, "%.3f\n", (double)(age1.total_bits + 40) / (Width  * Height));
		writeBitRate(buffer, FILEName, size);
#endif
		
#endif

#if ADAPTIVE_ARITHMETIC_CODING
		//Arithmatic
		printf("Bits : %d\n", ac_encoder_bits(&ace1));
		printf("Bit Rate : %7.3f\n", (double)(ac_encoder_bits(&ace1) + 40) / (Width  * Height));
		ac_encoder_done(&ace1);
		ac_model_done(&acm1);
		ac_model_done(&acm2);
		ac_model_done(&acm3);
		ac_model_done(&acm4);
		
		ac_model_done(&acm5);
		ac_model_done(&acm6);

		ac_model_done(&acm7);
		ac_model_done(&acm8);
		ac_model_done(&acm9);
		//*/
#endif
#if ADAPTIVE_GOLOMB_CODING
		//Golomb
		printf("Bits : %d\n", age1.total_bits);
		printf("Bit Rate : %7.3f\n", (double)(age1.total_bits + 40) / (Width  * Height));
		printf("k1 : %d\n", agm1.k);


		AG_encoder_done(&age1);
		AG_model_done(&agm1);

		//*/
#endif

#if OUTPUT_IMG
		/*
		for (int h = 0; h<Height; h++){
		for (int w = 0; w<Width; w++){
		int x = h * Width + w;
		if (*(ErrorImage + x) < -127){//G Channel <0
		image.at<Vec3b>(h, w) = Vec3b(0, 255, 0);//BGR
		printf("x:%d,y:%d,%d\n", w, h, *(ErrorImage + x));
		printf("a16: %f,%f,%f,%f,%f,%f\n", a16[0], a16[1], a16[2], a16[3], a16[4], a16[5]);
		}
		else if (*(ErrorImage + x) > 127){//R Channel >256
		image.at<Vec3b>(h, w) = Vec3b(0, 0, 255);//BGR
		printf("x:%d,y:%d,%d\n", w, h, *(ErrorImage + x));
		printf("a16: %f,%f,%f,%f,%f,%f\n", a16[0], a16[1], a16[2], a16[3], a16[4], a16[5]);
		}
		}
		}
		*/
		//gray_image.data = EdgeImage;
		//imwrite(OutImageName, image);//RGB 3 Channel Image 
		WriteImage(DeltaImage, gray_image.data);
		imwrite(OutImageName, gray_image);//Gray Image
#endif
#if OUTPUT_TIME

		//TranArea_time += (double)(TranArea_end - TranArea_start) / CLOCKS_PER_SEC;
		//TranSize_time += (double)(TranSize_end - TranSize_start) / CLOCKS_PER_SEC;
		//normal_time += (double)(normal_end - normal_start) / CLOCKS_PER_SEC;

		printf("TranSize Time :\t %3.1f s \n", TranSize_time);
		printf("Tranarea Time :\t %3.1f s \n", TranArea_time);
		printf("Normal Equation Time :\t %3.1f s \n", normal_time);
		/*
		compress_time = (double)(compress_end - compress_start2) / CLOCKS_PER_SEC;

		printf("LSR Time =\t %f (s)\n", compress_time);
		printf("Normal Equation Time =\t %f (s)\n", Tran_time);
		*/
		compress_time = (double)(compress_end - compress_start) / CLOCKS_PER_SEC;
		printf("Total Compress Time =\t %f (s)\n", compress_time);


		sprintf(timetext, "%d\t%.5f\t%.5f\t%.5f\n\r", predictor_order, TranSize_time, TranArea_time, normal_time);
		writetime(timetext, FILEName);
#endif
#if OUTPUT_COUNT
		printf("Cholesky Count: %d \t", (choleskyCount - svdCount));
		printf("SVD Count: %d\n", svdCount);
		sprintf(Counttext, "%d,%d  %d\n\r", predictor_order, (choleskyCount - svdCount), svdCount);
		writeCounttimes(Counttext, FILEName);
#endif
#if OUTPUT_DATA
		if (predictor_order == PredictorOrderSetting && SettingN){

			for (int i = 0; i<(Width)*(Height)*(Pixel_Byte); i++){
				sprintf((predictdata + i * 8), "%d\n", *(DeltaImage + i));
			}
			writedata(predictdata, FILEName);
		}
		else if (!SettingN){
			for (int i = 0; i<(Width)*(Height)*(Pixel_Byte); i++){
				sprintf((predictdata + i * 8), "%d\n", *(DeltaImage + i));
			}
			writedata(predictdata, FILEName);
		}
		//writedata(e, FILEName);
#endif
#if OUTPUT_INT_DATA
		if (predictor_order == PredictorOrderSetting && SettingN){
			writeintdata(DeltaImage, FILEName);
		}
		else if (!SettingN){
			writeintdata(DeltaImage, FILEName);
		}
		//writedata(e, FILEName);
#endif
#if OUTPUT_UCHAR_DATA
		if (predictor_order == PredictorOrderSetting && SettingN){
			writeUCdata(DeltaImage, FILEName);
		}
		else if (!SettingN){
			writeUCdata(DeltaImage, FILEName);
		}
		//writedata(e, FILEName);
#endif
#if OUTPUT_ENTROPY
		int len, *hist, histlen;
		double H;
		len = (Width)*(Height);
		hist = (int*)calloc(len, sizeof(int));
		histlen = makehist(DeltaImage, hist, len);
		//histlen = makehist(ErrorImage, hist, len);
		//hist now has no order (known to the program) but that doesn't matter
		H = entropy(hist, histlen, len);
		//printf("%lf\n",H);
		sprintf(entropytext, "%.5f\t", H);
		writeEntropy(entropytext, FILEName);
#endif
#if OUTPUT_CONTEXT_SD
		StandardDeviation();
		char buffer[50];
		for (int i = 0; i<ContextArraySize; i++){
			if (times_count[i] != 0){
				int size = sprintf(buffer, "Index:%d, \tTimes:%d, \tAvg:%.2f, \tVar:%.2f \n", i, times_count[i], context_avg[i], context_var[i]);
				writeContextSD(buffer, FILEName, size);
			}
		}
#endif

		free(SourceImage);
		free(ErrorImage);
		free(EdgeImage);
		free(predictdata);
		free(timetext);
		free(Counttext);
		free(e);
		free(entropytext);
		//printf("Count:%d\n",Count);
		printf("-------------------ENCODING END---------------------------\n");
	}
	else if (argc > 8) {
		printf("Too many arguments supplied.\nFILEName\tWidth\tHeight\tpredictor_order\t\n");
	}
	else {
		printf("One argument expected.\nFILEName\tWidth\tHeight\tpredictor_order\t\n");
	}
#if TEST_ONE	
	printf("Count1 : %d \n", Count1);
	printf("Count2 : %d \n", Count2);
	printf("Count3 : %d \n", Count3);
	printf("Count4 : %d \n", Count4);
	printf("Count5 : %d \n", Count5);
	printf("Count6 : %d \n", Count6);
	printf("Count7 : %d \n", Count7);
	printf("Count8 : %d \n", Count8);
	printf("Count9 : %d \n", Count9);
	system("pause");
#endif
	return 0;
}