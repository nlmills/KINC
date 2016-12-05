#include "mutualinfo.h"



void mutualinfo::input(DataPlugin* input)
{
   bool cond {input->type()==std::string("ematrix")};
   assert<InvalidInputType>(cond,__FILE__,__LINE__);
   _in.push_back(input);
}



void mutualinfo::output(DataPlugin* output)
{
   assert<TooManyOutputs>(_out==nullptr,__FILE__,__LINE__);
   _out = output;
}



void mutualinfo::execute_cl(GetOpts& ops, Terminal& tm)
{
   ;
}



void mutualinfo::execute_pn(GetOpts& ops, Terminal& tm)
{
   ;
}