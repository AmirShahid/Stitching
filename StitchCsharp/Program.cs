using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Runtime.InteropServices;
using System.IO;

namespace StitchCsharp
{
    class Program
    {
        [DllImport("Test.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Sum(int A, int B);
        //[DllImport("Test.dll", CallingConvention = CallingConvention.Cdecl)]
        //public static extern int read_image(string path);

        //[DllImport("Test.dll", CallingConvention = CallingConvention.Cdecl)]
        ////public static extern IntPtr get_lr_shifts(int A, int B, int C);
        //public static extern IntPtr GetArray(double A, double B);
        public static byte[] ImageToByteArray(System.Drawing.Image imageIn)
        {
            using (var ms = new MemoryStream())
            {
                imageIn.Save(ms, imageIn.RawFormat);
                return ms.ToArray();
            }
        }
        static void Main(string[] args)
        {
            var a = Sum(2,5);
            //Stitching.StitchWrapper s = new Stitching.StitchWrapper();
            //byte[][] data = new byte[2][];
            //Image a = Image.FromFile(@"E:\lamel_stitching\whole_lamel_data_5\img_0_6.jpeg");
            //Image b = Image.FromFile(@"E:\lamel_stitching\whole_lamel_data_5\img_0_7.jpeg");
            //data[0] = ImageToByteArray(a);
            //data[1] = ImageToByteArray(b);
            //double[] res = s.CalculateStitchLR(data);
        }
    }
}
