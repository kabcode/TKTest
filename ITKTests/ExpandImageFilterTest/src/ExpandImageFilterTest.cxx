#include "itkTileImageFilter.h"

static constexpr unsigned int Dim3D = 3;
static constexpr unsigned int Dim2D = 2;

using PixelType = double;
using Image2DType = itk::Image<PixelType, Dim2D>;
using Image3DType = itk::Image<PixelType, Dim3D>;

using TileFilterType = itk::TileImageFilter<Image2DType, Image3DType>;
TileFilterType::Pointer GetTileFilterPointer();
void CreateImage(Image2DType::Pointer);

int main( int argc, char* argv[])
{
	std::vector<Image3DType::Pointer> Vector3DImages;

	auto Filter1 = GetTileFilterPointer();
	auto Image = Image2DType::New();
	CreateImage(Image);
	Filter1->SetInput(0, Image);
	Filter1->Update();
	auto Image3D = Filter1->GetOutput();
	Vector3DImages.push_back(Image3D);
	Filter1 = nullptr;

	auto Filter2 = GetTileFilterPointer();
	auto Image2 = Image2DType::New();
	CreateImage(Image2);
	Filter2->SetInput(0, Image2);
	Filter2->Update();
	auto Image3D2 = Filter2->GetOutput();
	Vector3DImages.push_back(Image3D2);
	Filter2 = nullptr;

	auto Filter3 = GetTileFilterPointer();
	Filter3->SetInput(0, Image);
	Filter3->SetInput(1, Image2);
	Filter3->Update();
	auto Image3D3 = Filter3->GetOutput();
	Vector3DImages.push_back(Image3D3);

	for(auto i = 0;i < Vector3DImages.size(); ++i)
	{
		Vector3DImages[i].Print(std::cout);
	}

	return EXIT_SUCCESS;
}

TileFilterType::Pointer GetTileFilterPointer()
{
	auto TileFilter = TileFilterType::New();
	itk::FixedArray<unsigned int, 3> Layout;
	Layout[0] = 1;
	Layout[1] = 1;
	Layout[2] = 0;
	TileFilter->SetLayout(Layout);

	return TileFilter;
}
void CreateImage(Image2DType::Pointer Image)
{
	Image2DType::IndexType idx;
	idx.Fill(0);

	Image2DType::SizeType sz;
	sz.Fill(100);

	Image2DType::RegionType rg(idx, sz);
	Image->SetRegions(rg);
	Image->Allocate();
	Image->FillBuffer(100);
}