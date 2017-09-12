#include "amatrix.h"
#include "cmatrix.h"



AMatrix::AMatrix():
   Node(sizeof(Header))
{
   init_data<Header>();
}



void AMatrix::init()
{
   try
   {
      // Copy file object from File
      Node::mem(File::mem());
      // If this is an empty and new file object...
      if (File::head()==fnullptr)
      {
         // Allocate space for headers, set File memory address, initialize header data to a null
         // state, and lastly write to file memory.
         allocate();
         File::head(addr());
         null_data();
         write();
      }
      else
      {
         // Grab header address of file memory and read it.
         addr(File::head());
         read();
         // Read in all gene names.
         Ace::FString fstr(File::mem(),data()._genePtr);
         fstr.static_buffer(_strSize);
         _geneNames.push_back(fstr.str());
         for (int i=1;i<data()._geneSize;++i)
         {
            fstr.bump();
            _geneNames.push_back(fstr.str());
         }
         // Set this object as NOT new.
         _isNew = false;
      }
   }
   catch (...)
   {
      // If any exception occurs while trying to load from file memory, clear all file memory and
      // set object to new.
      File::clear();
      _isNew = true;
      throw;
   }
}



void AMatrix::load(Ace::GetOpts &ops, Ace::Terminal &tm)
{
   tm << "Not yet implemented.\n";
}



void AMatrix::dump(Ace::GetOpts &ops, Ace::Terminal &tm)
{
   tm << "Not yet implemented.\n";
}



void AMatrix::query(Ace::GetOpts &ops, Ace::Terminal &tm)
{
   tm << "Not yet implemented.\n";
}



bool AMatrix::empty()
{
   return _isNew;
}



void AMatrix::initialize(std::vector<std::string>&& geneNames)
{
   // initialize.
   static const char* f = __PRETTY_FUNCTION__;
   // make sure sizes of vectors are valid and data object is empty.
   Ace::assert<InvalidSize>(geneNames.size()>0,f,__LINE__);
   if (File::head()==fnullptr)
   {
      addr(fnullptr);
      allocate();
      File::head(addr());
   }
   else
   {
      Ace::assert<AlreadySet>(_isNew,f,__LINE__);
   }
   try
   {
      // initialize header info.
      data()._geneSize = geneNames.size();
      Ace::FString fstr(File::mem());
      // initialize gene names.
      auto i = geneNames.begin();
      data()._genePtr = fstr.write(*i);
      ++i;
      while (i!=geneNames.end())
      {
         fstr.reset();
         fstr.write(*i);
         ++i;
      }
      // initialize edge data.
      data()._netData = Iterator::initialize(File::rmem(),geneNames.size());
      // write header info to file.
      write();
      // move vectors of gene names to internal objects.
      _geneNames = std::move(geneNames);
      _isNew = false;
   }
   catch (...)
   {
      // If any exception is thrown while initializing, make object empty.
      File::head(fnullptr);
      null_data();
      _isNew = true;
   }
}



AMatrix::Iterator AMatrix::begin()
{
   static const char* f = __PRETTY_FUNCTION__;
   Ace::assert<NullMatrix>(!_isNew,f,__LINE__);
   return Iterator(this,1,0);
}



AMatrix::Iterator AMatrix::end()
{
   static const char* f = __PRETTY_FUNCTION__;
   Ace::assert<NullMatrix>(!_isNew,f,__LINE__);
   return Iterator(this,data()._geneSize,0);
}



AMatrix::Iterator& AMatrix::at(int x, int y)
{
   static const char* f = __PRETTY_FUNCTION__;
   Ace::assert<NullMatrix>(!_isNew,f,__LINE__);
   Ace::assert<OutOfRange>(x>=0&&y>=0&&y<x&&x<data()._geneSize,f,__LINE__);
   return ref(x,y);
}



AMatrix::Iterator& AMatrix::ref(int x, int y)
{
   if (!_iterator.get())
   {
      _iterator.reset(new Iterator(this,x,y));
   }
   else
   {
      _iterator->set(x,y);
   }
   return *_iterator;
}



void AMatrix::flip_endian()
{
   flip(0,4);
   flip(4,8);
   flip(12,8);
}



void AMatrix::Iterator::read()
{
   Node::read(CMatrix::diagonal(_x,_y));
}



void AMatrix::Iterator::write()
{
   Node::write(CMatrix::diagonal(_x,_y));
}



int AMatrix::Iterator::x() const
{
   return _x;
}



int AMatrix::Iterator::y() const
{
   return _y;
}



uint8_t& AMatrix::Iterator::operator*()
{
   return get<uint8_t>();
}



bool AMatrix::Iterator::operator!=(const Iterator& cmp)
{
   return _p!=cmp._p||_x!=cmp._x||_y!=cmp._y;
}



void AMatrix::Iterator::operator++()
{
   if (_x<(_p->data()._geneSize))
   {
      if ((_y+1)<_x)
      {
         _y++;
      }
      else
      {
         _y = 0;
         ++_x;
      }
   }
}



AMatrix::Iterator::Iterator(AMatrix* p, int x, int y):
   Node(sizeof(uint8_t),p->File::mem(),p->data()._netData),
   _p(p),
   _x(x),
   _y(y)
{}



void AMatrix::Iterator::set(int x, int y)
{
   bool change {_x!=x||_y!=y};
   _x = x;
   _y = y;
   if (change)
   {
      read();
   }
}



int64_t AMatrix::Iterator::initialize(Ace::NVMemory& mem, int geneSize)
{
   return mem.allocate(sizeof(uint8_t)*CMatrix::diag_size(geneSize));
}