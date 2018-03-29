/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 *
 * Author: Kien le Trung trung-kien.le@eurecom.fr
 */

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

short sign(short x) {
  return (x > 0) - (x < 0);
}

short *ldpc_decoder(short *msgChannel,short block_length,short No_iteration,double rate) {
  // input
  //short No_iteration;
  //int *shift_value, *Col_position, *no_one_element, Z, BG,Kb;
  //double rate;
  //output
  //char *output_estimate;
  short *v_estimate;      //estimated codeword
  //variables
  //int *shift_value, *Col_position, *no_one_element, Zc, BG,Kb;
  int Zc, BG,Kb;
  int nrows, ncols, nEdge_base_graph;
  int irow, iShift, iZ,p1,p2,t1,temp_row,temp_col,iBit,iEdge;
  int nEdge,nCheck,nBit;
  int rows_total_La1=0,rows_total_La2=0,rows_total_La3=0,*no_rows_La1, *no_rows_La2, *no_rows_La3;
  int i1, i2, i3, sum, sum1, sum2, no_punctured_columns, layer, iLayer, n=1, flag=0;
  int *idxBit, idxCheck,*degBit, *degCheck,*degBit_base_graph, *pointerCheck, *pointerBit, *pointerCheck_temp, *pointerBit_temp;
  int *idxEdge2Bit,*idxEdge2Check;
  int *no_rows;
  int rows_total;
  short *msgBit2Check, *msgCheck2Bit, *msgBit;  //variables for message passing algorithm
  short mini;
  short sgn;
  int isEqual;
  char syn;      //syndrome
  short lift_size[51]= {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,64,72,80,88,96,104,112,120,128,144,160,176,192,208,224,240,256,288,320,352,384};
  short shift_value[197]= {9,117,204,26,189,205,0,0,167,166,253,125,226,156,224,252,0,0,81,114,44,52,240,1,0,0,8,58,158,104,209,54,18,128,0,0,179,214,71,0,231,41,194,159,103,0,155,228,45,28,158,0,129,147,140,3,116,0,142,94,230,0,203,205,61,247,0,11,185,0,117,0,11,236,210,56,0,63,111,14,0,83,2,38,222,0,115,145,3,232,0,51,175,213,0,203,142,8,242,0,254,124,114,64,0,220,194,50,0,87,20,185,0,26,105,29,0,76,42,210,0,222,63,0,23,235,238,0,46,139,8,0,228,156,0,29,143,160,122,0,8,151,0,98,101,135,0,18,28,0,71,240,9,84,0,106,1,0,242,44,166,0,132,164,235,0,147,85,36,0,57,40,63,0,140,38,154,0,219,151,0,31,66,38,0,239,172,34,0,0,75,120,0,129,229,118,0};
  short Col_position[197]= {0,1,2,3,6,9,10,11,0,3,4,5,6,7,8,9,11,12,0,1,3,4,8,10,12,13,1,2,4,5,6,7,8,9,10,13,0,1,11,14,0,1,5,7,11,15,0,5,7,9,11,16,1,5,7,11,13,17,0,1,12,18,1,8,10,11,19,0,1,6,7,20,0,7,9,13,21,1,3,11,22,0,1,8,13,23,1,6,11,13,24,0,10,11,25,1,9,11,12,26,1,5,11,12,27,0,6,7,28,0,1,10,29,1,4,11,30,0,8,13,31,1,2,32,0,3,5,33,1,2,9,34,0,5,35,2,7,12,13,36,0,6,37,1,2,5,38,0,4,39,2,5,7,9,40,1,13,41,0,5,12,42,2,7,10,43,0,12,13,44,1,5,11,45,0,2,7,46,10,13,47,1,5,11,48,0,7,12,49,2,10,13,50,1,5,11,51};
  short no_one_element[42]= {8, 10, 8, 10, 4, 6, 6, 6, 4, 5, 5, 5, 4, 5, 5, 4, 5, 5, 4, 4, 4, 4, 3, 4, 4, 3, 5, 3, 4, 3, 5, 3, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4};

  if (block_length>3840)
  {
    BG=1;
    Kb = 22;
    nrows=46;
    ncols=68;
  }
  else if (block_length<=3840)
  {
    BG=2;
    nrows=42;

    if (block_length>640)
    {
      Kb = 10;
      ncols=52;
    }
    else if (block_length>560)
    {
      Kb = 9;
      ncols=51;
    }
    else if (block_length>192)
    {
      Kb = 8;
      ncols=50;
    }
    else
    {
      Kb = 6;
      ncols=48;
    }
  }

  //find minimum value in all sets of lifting size
  for (i1=0; i1 < 51; i1++)
  {
    if (lift_size[i1] >= (double) block_length/Kb)
    {
      Zc = lift_size[i1];
      // printf("%d",Zc);
      break;
    }
  }

  nEdge_base_graph=197;
  nEdge=nEdge_base_graph*Zc;
  nCheck=nrows*Zc;
  nBit=ncols*Zc;
  //initial positions of pointers to check nodes
  //degrees of check nodes
  pointerCheck = (int *)malloc(sizeof(int) * nCheck);
  degCheck=(int *)malloc(sizeof(int) * nCheck);

  for (i1=0,temp_row=0; i1 < nrows; i1++)
  {
    for (i2=0; i2 < Zc; i2++)
    {
      degCheck[i1*Zc+i2] = no_one_element[i1];   //degree equals number of 1 elements in a row
      pointerCheck[i1*Zc+i2] = temp_row;
      temp_row = temp_row + no_one_element[i1];
    }
  }

  //initial positions of pointers to bit nodes
  //degrees of bit nodes
  pointerBit = (int *)malloc(sizeof(int) * nBit);
  degBit=(int *)malloc(sizeof(int) * nBit);
  degBit_base_graph=(int *)malloc(sizeof(int) * ncols);
  memset(degBit_base_graph,0,sizeof(int) * ncols);

  for (iBit=0; iBit < nEdge_base_graph; ++iBit)
  {
    ++degBit_base_graph[Col_position[iBit]];    //number of 1 elements in a columns in base graph
  }

  for (i1=0,temp_col=0; i1 < ncols; i1++)
  {
    for (i2=0; i2 < Zc; i2++)
    {
      degBit[i1*Zc+i2] = degBit_base_graph[i1];   //degree equals number of 1 elements in a column
      pointerBit[i1*Zc+i2] = temp_col;
      temp_col = temp_col + degBit_base_graph[i1];
    }
  }

  //indice and degrees of check nodes and bit nodes
  //divide layer for message passing algorithm
  idxBit=(int *)malloc(sizeof(int) * nEdge);
  // idxCheck=mxMalloc(sizeof(int) * nEdge);
  idxEdge2Bit=(int *)malloc(sizeof(int) * nEdge);
  idxEdge2Check=(int *)malloc(sizeof(int) * nEdge);
  pointerCheck_temp = (int *)malloc(sizeof(int) * nCheck);
  memcpy(pointerCheck_temp,pointerCheck,sizeof(int) * nCheck);
  pointerBit_temp = (int *)malloc(sizeof(int) * nBit);
  memcpy(pointerBit_temp,pointerBit,sizeof(int) * nBit);
  no_rows_La1=(int *)malloc(sizeof(int) * nBit);
  no_rows_La2=(int *)malloc(sizeof(int) * nBit);
  no_rows_La3=(int *)malloc(sizeof(int) * nBit);

  for (irow=0,p1=0,t1=0,iEdge=0; irow < nrows; ++irow)   //loop for rows in base graph
  {
    temp_row=irow*Zc;
    sum=0; sum1=0; sum2=0;

    for (iShift=0; iShift < no_one_element[irow]; ++iShift)  //loop for 1 elements in one row of base graph
    {
      temp_col=Col_position[p1]*Zc;

      if ( ((rate==0.2)&&(BG==2)&&(Kb==10)) || ((rate==0.33)&&(BG==1)&&(Kb==22)) )   //layer , no rate matching
      {
        layer=2;

        if (Col_position[p1]==0||Col_position[p1]==1)
          sum++;
      }
      else if ( ( (BG==1) && (rate>=0.33) && (rate<=0.89) )||( (BG==2) && (rate>=0.2) && (rate<=0.67) ) ) //layer, rate matching
      {
        layer=3;
        no_punctured_columns=ceil(nBit/Zc-2-Kb/rate);

        if (Col_position[p1]==0 || Col_position[p1]==1)
          sum1++;

        if ( (Col_position[p1] >= ncols-no_punctured_columns) && (Col_position[p1] < ncols) )
          sum2++;
      }

      for (iZ=0,p2=0; iZ < Zc; iZ++)       //loop for lift size
      {
        idxBit[t1] = (shift_value[p1]+p2)%Zc + temp_col;     //column positions
        p2++;
        idxCheck = temp_row + iZ;                       // row positions
        idxEdge2Check[ pointerCheck_temp[idxCheck] ] = iEdge;  //label and store the edges connecting to check nodes
        ++pointerCheck_temp[idxCheck];
        idxEdge2Bit[ pointerBit_temp[idxBit[t1]] ] = iEdge;   //label and store the edges connecting to bit nodes
        ++pointerBit_temp[idxBit[t1]];
        iEdge++;
        t1++;
      }

      p1++;
    }

    if ( ((rate==0.2)&&(BG==2)&&(Kb==10)) || ((rate==0.33)&&(BG==1)&&(Kb==22)) )
    {
      if (sum==1||sum==0)
      {
        for (i1=0; i1 < Zc; i1++)
        {
          no_rows_La1[rows_total_La1]=irow*Zc + i1;
          rows_total_La1++;
        }
      }
      else
      {
        for (i1=0; i1 < Zc; i1++)
        {
          no_rows_La2[rows_total_La2]=irow*Zc + i1;
          rows_total_La2++;
        }
      }
    }
    else if (( (BG==1) && (rate>=0.33) && (rate<=0.89) )||( (BG==2) && (rate>=0.2) && (rate<=0.67) ))
    {
      sum=sum1+sum2;

      if (sum==1)
      {
        for (i1=0; i1 < Zc; i1++)
        {
          no_rows_La1[rows_total_La1]=irow*Zc + i1;
          rows_total_La1++;
        }
      }
      else if (sum==2)
      {
        if (sum1==1)
        {
          for (i1=0; i1 < Zc; i1++)
          {
            no_rows_La2[rows_total_La2]=irow*Zc + i1;
            rows_total_La2++;
          }
        }
        else if (sum1==2)
        {
          for (i1=0; i1 < Zc; i1++)
          {
            no_rows_La3[rows_total_La3]=irow*Zc + i1;
            rows_total_La3++;
          }
        }
      }
      else if (sum==3)
      {
        for (i1=0; i1 < Zc; i1++)
        {
          no_rows_La2[rows_total_La2]=irow*Zc + i1;
          rows_total_La2++;
        }
      }
      else if (sum==0)
      {
        for (i1=0; i1 < Zc; i1++)
        {
          no_rows_La3[rows_total_La3]=irow*Zc + i1;
          rows_total_La3++;
        }
      }
    }
  }

  // allocate memory for message passing algorithm
  msgBit2Check=(short *)malloc(sizeof(short) * nEdge);
  msgCheck2Bit=(short *)malloc(sizeof(short) * nEdge);
  msgBit=(short *)malloc(sizeof(short) * nBit);
  memset(msgCheck2Bit,0,sizeof(short) * nEdge);
  v_estimate=(short *)malloc(sizeof(short) * nBit);
  // initial values of LLR of bit nodes
  memcpy(msgBit,msgChannel,nBit*sizeof(short));

  //message passing algorithm
  while (n<=No_iteration)
  {
    for (iLayer=1; iLayer<=layer; iLayer++)
    {
      if (iLayer==1)
      {
        no_rows=no_rows_La1;
        rows_total=rows_total_La1;
      }
      else if (iLayer==2)
      {
        no_rows=no_rows_La2;
        rows_total=rows_total_La2;
      }
      else if (iLayer==3)
      {
        no_rows=no_rows_La3;
        rows_total=rows_total_La3;
      }

      //message from bit nodes to check nodes
      for(i1 = 0; i1 < rows_total; ++i1)
      {
        for(i2 = 0; i2 < degCheck[no_rows[i1]]; ++i2)
        {
          msgBit2Check[ idxEdge2Check[ pointerCheck[no_rows[i1]] + i2] ] = msgBit[ idxBit[ idxEdge2Check[ pointerCheck[no_rows[i1]] + i2] ] ]
              -msgCheck2Bit[ idxEdge2Check[ pointerCheck[no_rows[i1]] + i2] ];
        }
      }

      //message from check nodes to bit nodes
      for(i1 = 0; i1 < rows_total; ++i1)
      {
        for(i2 = 0; i2 < degCheck[no_rows[i1]]; ++i2)
        {
          mini=32640;
          sgn=1;

          for(i3 = 0; i3 < degCheck[no_rows[i1]]; ++i3)
          {
            if (idxEdge2Check[ pointerCheck[no_rows[i1]] + i3] != idxEdge2Check[ pointerCheck[no_rows[i1]] + i2 ])
            {
              sgn *=sign(msgBit2Check[ idxEdge2Check[ pointerCheck[no_rows[i1]] + i3] ]);

              if (abs(msgBit2Check[ idxEdge2Check[ pointerCheck[no_rows[i1]] + i3] ]) < mini)
                mini=abs(msgBit2Check[ idxEdge2Check[ pointerCheck[no_rows[i1]] + i3] ]);
            }
          }

          msgCheck2Bit[ idxEdge2Check[ pointerCheck[no_rows[i1]] + i2 ] ]=sgn * mini;

          //atanh(1)=19.07, 19.07 is converted to fixed-point 9_7= 19.07*2^7=2441
          if ( msgCheck2Bit[ idxEdge2Check[ pointerCheck[no_rows[i1]] + i2 ] ] > 2441)
            msgCheck2Bit[ idxEdge2Check[ pointerCheck[no_rows[i1]] + i2 ] ] = 2441;

          if (msgCheck2Bit[ idxEdge2Check[ pointerCheck[no_rows[i1]] + i2 ] ] < -2441)
            msgCheck2Bit[ idxEdge2Check[ pointerCheck[no_rows[i1]] + i2 ] ] = -2441;
        }
      }

      // LLR
      for(iBit = 0; iBit < nBit; ++iBit)
      {
        msgBit[iBit]=msgChannel[iBit];

        for(i1 = 0; i1 < degBit[iBit]; ++i1)
          msgBit[iBit] += msgCheck2Bit[idxEdge2Bit[ pointerBit[iBit]+i1 ] ];

        if (msgBit[iBit]>=0)
          v_estimate[iBit]=0;
        else
          v_estimate[iBit]=1;
      }

      //     // LLR
      //     for(iBit = 0; iBit < nBit; ++iBit)
      //     {
      //         msgBit[iBit]=msgChannel[iBit];
      //         for(i1 = 0; i1 < degBit[iBit]; ++i1)
      //             msgBit[iBit] += msgCheck2Bit[idxEdge2Bit[ pointerBit[iBit]+i1 ] ];
      //     }
      //
      //     //estimate codeword
      //     v_estimate=mxMalloc(sizeof(char) * nBit);
      //     for(iBit = 0; iBit < nBit; ++iBit)
      //     {
      //         if (msgBit[iBit]>=0)
      //             v_estimate[iBit]=0;
      //         else
      //             v_estimate[iBit]=1;
      //     }

      //check syndrome=0
      for (i1=0,isEqual=1; i1 < nCheck; ++i1)
      {
        syn=0;

        for(i2 = 0; i2 < degCheck[i1]; ++i2)
        {
          //sum of bits of estimated codeword in the positions where there are 1 elements in graph
          syn += v_estimate[ idxBit[ idxEdge2Check[ pointerCheck[ i1 ] + i2  ] ] ];
        }

        syn=syn%2;

        if (syn != 0)
        {
          isEqual=0;
          break;
        }
      }

      if (isEqual==1)   //syndrome=0, break
      {
        flag=1;
        break;
      }
    }

    if (flag==1)
    {
      break;
    }

    n++;
  }

  return v_estimate;
}
