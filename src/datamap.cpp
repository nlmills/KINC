#include "datamap.h"



bool DataMap::add(const std::string& name, DataPlugin* data)
{
   bool ret = false;
   if (find(name)==_map.end())
   {
      _map[name] = data;
      ret = true;
   }
   return ret;
}



bool DataMap::del(const std::string& name)
{
   return _map.erase(name)==1;
}



DataMap::StdMap::iterator DataMap::find(const std::string& name)
{
   return _map.find(name);
}



DataMap::StdMap::iterator DataMap::begin()
{
   return _map.begin();
}



DataMap::StdMap::iterator DataMap::end()
{
   return _map.end();
}
