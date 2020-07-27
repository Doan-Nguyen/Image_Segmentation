+ Encoder: (CNN)
    - giam chieu rong & dai image by using conv & pooling

+ Decoder: (FCN)
    - phuc hoi kich thuoc anh ban dau 
    - FCN 8, 16, 32

+ What is the difference between upsampling and bi-linear upsampling in a CNN?

    - Upsampling: increasing the size of an image

    - Upsampling techniques:

        - Nearest-Neighbor: copies the value from nearest pixel's value

        - Bilinear: Uses all nearby pixels to calculate the pixel's value. Using **linear interpolations**

        - Bicubic: Same *Bilinear* but using **polynomial interpolations**

    - Details:

        - https://datascience.stackexchange.com/questions/38118/what-is-the-difference-between-upsampling-and-bi-linear-upsampling-in-a-cnn

+ Classmethod & Staticmethod in Python

    - *@classmethod*
