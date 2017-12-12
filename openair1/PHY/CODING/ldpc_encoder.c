#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "Gen_shift_value.h"

short *choose_generator_matrix(short BG,short Zc)
{
    short *Gen_shift_values;

    if (BG==1)
    {
        switch (Zc)
        {
            case 2: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_2;
                break;
            case 3: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_3;
                break;
            case 4: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_4;
                break;
            case 5: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_5;
                break;
            case 6: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_6;
                break;
            case 7: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_7;
                break;
            case 8: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_8;
                break;
            case 9: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_9;
                break;
            case 10: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_10;
                break;
            case 11: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_11;
                break;
            case 12: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_12;
                break;
            case 13: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_13;
                break;
            case 14: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_14;
                break;
            case 15: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_15;
                break;
            case 16: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_16;
                break;
            case 18: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_18;
                break;
            case 20: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_20;
                break;
            case 22: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_22;
                break;
            case 24: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_24;
                break;
            case 26: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_26;
                break;
            case 28: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_28;
                break;
            case 30: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_30;
                break;
            case 32: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_32;
                break;
            case 36: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_36;
                break;
            case 40: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_40;
                break;
            case 44: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_44;
                break;
            case 48: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_48;
                break;
            case 52: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_52;
                break;
            case 56: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_56;
                break;
            case 60: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_60;
                break;
            case 64: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_64;
                break;
            case 72: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_72;
                break;
            case 80: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_80;
                break;
            case 88: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_88;
                break;
            case 96: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_96;
                break;
            case 104: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_104;
                break;
            case 112: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_112;
                break;
            case 120: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_120;
                break;
            case 128: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_128;
                break;
            case 144: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_144;
                break;
            case 160: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_160;
                break;
            case 176: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_176;
                break;
            case 192: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_192;
                break;
            case 208: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_208;
                break;
            case 224: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_224;
                break;
            case 240: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_240;
                break;
            case 256: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_256;
                break;
            case 288: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_288;
                break;
            case 320: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_320;
                break;
            case 352: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_352;
                break;
            case 384: Gen_shift_values=(short*) Gen_shift_values_BG1_Z_384;
                break;
        }
    }
/*
    else if (BG==2)
    {
            case 2: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_2;
                break;
            case 3: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_3;
                break;
            case 4: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_4;
                break;
            case 5: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_5;
                break;
            case 6: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_6;
                break;
            case 7: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_7;
                break;
            case 8: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_8;
                break;
            case 9: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_9;
                break;
            case 10: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_10;
                break;
            case 11: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_11;
                break;
            case 12: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_12;
                break;
            case 13: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_13;
                break;
            case 14: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_14;
                break;
            case 15: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_15;
                break;
            case 16: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_16;
                break;
            case 18: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_18;
                break;
            case 20: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_20;
                break;
            case 22: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_22;
                break;
            case 24: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_24;
                break;
            case 26: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_26;
                break;
            case 28: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_28;
                break;
            case 30: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_30;
                break;
            case 32: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_32;
                break;
            case 36: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_36;
                break;
            case 40: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_40;
                break;
            case 44: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_44;
                break;
            case 48: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_48;
                break;
            case 52: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_52;
                break;
            case 56: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_56;
                break;
            case 60: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_60;
                break;
            case 64: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_64;
                break;
            case 72: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_72;
                break;
            case 80: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_80;
                break;
            case 88: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_88;
                break;
            case 96: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_96;
                break;
            case 104: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_104;
                break;
            case 112: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_112;
                break;
            case 120: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_120;
                break;
            case 128: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_128;
                break;
            case 144: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_144;
                break;
            case 160: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_160;
                break;
            case 176: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_176;
                break;
            case 192: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_192;
                break;
            case 208: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_208;
                break;
            case 224: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_224;
                break;
            case 240: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_240;
                break;
            case 256: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_256;
                break;
            case 288: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_288;
                break;
            case 320: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_320;
                break;
            case 352: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_352;
                break;
            case 384: Gen_shift_values=(short*) Gen_shift_values_BG2_Z_384;
                break;
    }
*/
    return Gen_shift_values;
}

int ldpc_encoder(char *test_input,char* channel_input,short block_length,double rate)
	{
		char c[22*384]; //padded input, unpacked, max size
		char d[68*384]; //coded output, unpacked, max size 
		short *Gen_shift_values, *no_shift_values, *pointer_shift_values;
		short BG,Zc,Kb,nrows,ncols,channel_temp;
		int i,i1,i2,i3,i4,i5,t,temp,temp_prime;
		int no_punctured_columns,output_length;  

		//Table of possible lifting sizes
		short lift_size[51]={2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,64,72,80,88,96,104,112,120,128,144,160,176,192,208,224,240,256,288,320,352,384};
	
		 //determine number of bits in codeword
		 if (block_length>3840)
		   {
			  BG=1;
			  Kb = 22;
			  nrows=46;	//parity check bits
			  ncols=22;	//info bits
		   }
		 else if (block_length<=3840)
		   {
			  BG=2;
			  nrows=42;	//parity check bits
			  ncols=10;	// info bits
			  if (block_length>640)
				 Kb = 10;
			  else if (block_length>560)
				 Kb = 9;
			  else if (block_length>192)
				 Kb = 8;
			  else
				 Kb = 6;
		   }

		 //find minimum value in all sets of lifting size
		 for (i1=0; i1 < 51; i1++)
		   {
			  if (lift_size[i1] >= (double) block_length/Kb)
				{
				  Zc = lift_size[i1];
				   //printf("%d\n",Zc);
				  break;
				}
		   }

		 // load base graph of generator matrix
		 if (BG==1)
		   {
			  no_shift_values=(short*) no_shift_values_BG1;
			  pointer_shift_values=(short*) pointer_shift_values_BG1;          
		   }

		 else if (BG==2)
		   {       
		   no_shift_values=(short*) no_shift_values_BG2;
		   pointer_shift_values=(short*) pointer_shift_values_BG2;
			   
		   }
		 Gen_shift_values=choose_generator_matrix(BG,Zc);
		 no_punctured_columns=(int)((nrows+Kb-2)*Zc-block_length/rate)/Zc; 
		//printf("%d\n",no_punctured_columns);

		 output_length = (Kb+nrows-no_punctured_columns) * Zc;
		
		 // unpack input
		for (i=0;i<block_length;i++) 
		  c[i] = (test_input[i/8]>>(i%8))&1;
		
		// parity check part
		 for (i2=0; i2 < Zc; i2++)
		   {
			  t=Kb*Zc+i2;
			  //rotate matrix here
		  for (i5=0; i5 < Kb; i5++)
			{
			   temp = c[i5*Zc];
			   memmove(&c[i5*Zc], &c[i5*Zc+1], (Zc-1)*sizeof(short));
			   c[i5*Zc+Zc-1] = temp;           
			}   
			 
			  // calculate each row in base graph
			  for (i1=0; i1 < nrows-no_punctured_columns; i1++) 
			   {
			   channel_temp=0;
			   for (i3=0; i3 < Kb; i3++)
					 {
				temp_prime=i1 * ncols + i3;
						for (i4=0; i4 < no_shift_values[temp_prime]; i4++)
						  {
							 channel_temp = channel_temp ^ c[ i3*Zc + Gen_shift_values[ pointer_shift_values[temp_prime]+i4 ] ];
						  }
					 }

			   d[t+i1*Zc]=channel_temp;
			
			   }
		   }

		// information part
		memcpy(d,c,Kb*Zc*sizeof(short));

		//pack into output
		memset(channel_input,0,output_length/8*sizeof(char));
		for (i=0;i<output_length;i++)
		  channel_input[i/8] = channel_input[i/8] | (d[i] << (i%8));
		
		return 0;
	}
