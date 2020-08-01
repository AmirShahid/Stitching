using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.IO;

namespace CsharpTest
{
    [StructLayout(LayoutKind.Sequential)]
    public struct ImageFile
    {
        public IntPtr ByteArray;
        public int Length;
    };

    [StructLayout(LayoutKind.Sequential)]
    public struct ImageRow
    {
        public IntPtr Files;
        public int FileCount;
    };

    [StructLayout(LayoutKind.Sequential)]
    struct LamelImages
    {
        public IntPtr rows;
        public int row_count;
    };

    [StructLayout(LayoutKind.Sequential)]
    struct FullLamelLevels
    {
        public IntPtr lamel_images;
        public int zoom_level_count;
    };



    [StructLayout(LayoutKind.Sequential)]
    public struct ShiftArrayRow
    {
        public IntPtr columns;
        public int column_count;
    };

    [StructLayout(LayoutKind.Sequential)]
    public struct ShiftArray
    {
        public IntPtr rows;
        public int row_count;
    };

    [StructLayout(LayoutKind.Sequential)]

    struct FullLamelImages
    {
        public IntPtr full_lamel_image;
        public int Length;
    }
    struct FullLamelImage
    {
        public ImageFile image_file;
        public int x, y, z;
    };


    class Program
    {
        [DllImport("Stitching.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int get_lr_shifts(ImageRow images, double[] shift_r, double[] shift_c);

        [DllImport("Stitching.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern FullLamelImages stitch_all(LamelImages images, int[] best_column, ShiftArray shift_r, ShiftArray shift_c);
        static void Main(string[] args)
        {
            int column_count = 10;
            int row_count = 20;
            int[] best_column_for_ud = new int[row_count];
            double[][] shifts_r = new double[row_count][];
            double[][] shifts_c = new double[row_count][];
            ImageRow[] imageRows = new ImageRow[row_count];
            ShiftArrayRow[] shiftRArrayRows = new ShiftArrayRow[row_count];
            ShiftArrayRow[] shiftCArrayRows = new ShiftArrayRow[row_count];
            for (int i = 0; i < row_count; i++)
            {
                byte[][] FilesArray = new byte[column_count][];
                for (int j = 0; j < column_count; j++)
                {
                    /// Loading dataset to RAM
                    FilesArray[j] = System.IO.File.ReadAllBytes(@"E:\lamel_stitching\whole_lamel_data_5\img_" + i.ToString() + "_" + j.ToString() + ".jpeg");
                }
                GCHandle[] Handles = new GCHandle[column_count];
                ImageFile[] imageFiles = new ImageFile[column_count];
                for (int j = 0; j < column_count; j++)
                {
                    Handles[j] = GCHandle.Alloc(FilesArray[j], GCHandleType.Pinned);
                    IntPtr ByteArray = Handles[j].AddrOfPinnedObject();
                    imageFiles[j] = new ImageFile()
                    {
                        ByteArray = ByteArray,
                        Length = FilesArray[j].Length
                    };
                }
                GCHandle rowHandler = GCHandle.Alloc(imageFiles, GCHandleType.Pinned);
                imageRows[i] = new ImageRow()
                {
                    Files = rowHandler.AddrOfPinnedObject(),
                    FileCount = imageFiles.Length
                };
                shifts_r[i] = new double[column_count];
                shifts_c[i] = new double[column_count];
                best_column_for_ud[i] = get_lr_shifts(imageRows[i], shifts_r[i], shifts_c[i]);
                GCHandle handle = GCHandle.Alloc(shifts_r[i], GCHandleType.Pinned);
                shiftRArrayRows[i] = new ShiftArrayRow()
                {
                    columns = handle.AddrOfPinnedObject(),
                    column_count = shifts_r[i].Length
                };
                handle = GCHandle.Alloc(shifts_c[i], GCHandleType.Pinned);
                shiftCArrayRows[i] = new ShiftArrayRow()
                {
                    columns = handle.AddrOfPinnedObject(),
                    column_count = shifts_c[i].Length
                };

            }
            Console.WriteLine("Stitch arrays calculated, starting whole lamel Stitch...");
            GCHandle LamelHandler = GCHandle.Alloc(imageRows, GCHandleType.Pinned);
            LamelImages lamelImages = new LamelImages()
            {
                rows = LamelHandler.AddrOfPinnedObject(),
                row_count = imageRows.Length
            };
            GCHandle RHandle = GCHandle.Alloc(shiftRArrayRows, GCHandleType.Pinned);
            ShiftArray shift_r = new ShiftArray()
            {
                rows = RHandle.AddrOfPinnedObject(),
                row_count = shiftRArrayRows.Length
            };
            GCHandle CHandle = GCHandle.Alloc(shiftCArrayRows, GCHandleType.Pinned);
            ShiftArray shift_c = new ShiftArray()
            {
                rows = CHandle.AddrOfPinnedObject(),
                row_count = shiftCArrayRows.Length
            };
            Console.WriteLine("Horizental shifts are calculated! \n \n" +
                "Now Calculating whole lamel output ...");
            Console.ReadKey();
            var fullLamelImages = stitch_all(lamelImages, best_column_for_ud, shift_r, shift_c);
            //FullLamelImage CurrentImage = Marshal.PtrToStructure<FullLamelImage>(ImagePointer);
            var Ptr = fullLamelImages.full_lamel_image;

            string OutputPath = Environment.CurrentDirectory + "\\output";
            System.IO.Directory.CreateDirectory(OutputPath);

            for (int i = 0; i < fullLamelImages.Length; i++)
            {
                var CurrentImage = Marshal.PtrToStructure<FullLamelImage>(Ptr);
                byte[] FileArray = new byte[CurrentImage.image_file.Length];
                Marshal.Copy(CurrentImage.image_file.ByteArray, FileArray, 0, CurrentImage.image_file.Length);
                var Size = Marshal.SizeOf(CurrentImage);

                Ptr = new IntPtr(Ptr.ToInt64() + Size);
                string FilePath = OutputPath+$"\\{CurrentImage.z}\\{CurrentImage.x}";
                System.IO.Directory.CreateDirectory(FilePath);

                System.IO.File.WriteAllBytes($"{FilePath}\\{CurrentImage.y}.jpg", FileArray);

            }
            var FullLamelImage = Marshal.PtrToStructure<FullLamelImage>(fullLamelImages.full_lamel_image);    
        }
    }
}
