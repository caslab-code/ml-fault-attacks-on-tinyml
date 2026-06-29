#include "floatPrint.h"

/*Code for Big Endian*/
int FloatToByte_B(int index, unsigned char outbox[], float number){
    unsigned long d = *(unsigned long *)&number;

    outbox[index] = (d & 0xFF000000) >> 24;
    index++;

    outbox[index] = (d & 0x00FF0000) >> 16;
    index++;

    outbox[index] = (d & 0x0000FF00) >> 8;
    index++;

    outbox[index] = d & 0x000000FF;
    index++;
    return index;
}

/*This system is Little Endian*/
int FloatToByte(int index, unsigned char outbox[], float number)
{
    unsigned long d = *(unsigned long *)&number;

    outbox[index] = d & 0x000000FF;
    index++;

    outbox[index] = (d & 0x0000FF00) >> 8;
    index++;

    outbox[index] = (d & 0x00FF0000) >> 16;
    index++;

    outbox[index] = (d & 0xFF000000) >> 24;
    index++;
    return index;
}

int Int8ToByte(int index, unsigned char outbox[], int8_t number)
{
    outbox[index] = (unsigned char)number;
    index++;
    return index;
}

int ByteToInt8(int index, unsigned char inbox[], int8_t *number)
{   
    *number = (int8_t)inbox[index];
    index++;
    return index;
}
