#pragma once
#ifndef PYLIKE_NUMPY_H
#define PYLIKE_NUMPY_H

#include <iostream>  
#include <vector>  
#include <algorithm>  


namespace np
{
    namespace impl_
    {
        // 定义一个结构体，用于存储值和索引
        template <class T>
        struct IndexedValue {  
            T value;  
            int index;  
        
            // 构造函数  
            IndexedValue(T val, int idx) : value(val), index(idx) {}  
        
            // 比较函数，用于升序排序  
            bool operator<(const IndexedValue& other) const {  
                return value < other.value;  
            }  
        
            // 如果需要降序排序，可以定义一个比较函数或者重载>操作符  
        };  
        
        // 自定义比较函数，用于降序排序
        template <class T> 
        bool compareDesc(const IndexedValue<T>& a, const IndexedValue<T>& b) {  
            return a.value > b.value;  
        }
    }
      
    
    // 实现类似np.argsort的函数
    template <class T>
    std::vector<int> argsort(const std::vector<T>& vec, bool reverse = false) {  
        std::vector<impl_::IndexedValue<T>> indexedValues;  
    
        // 填充索引和值  
        for (size_t i = 0; i < vec.size(); ++i) {  
            indexedValues.emplace_back(vec[i], static_cast<int>(i));  
        }  
    
        // 根据值排序索引  
        if (reverse) {  
            std::sort(indexedValues.begin(), indexedValues.end(), impl_::compareDesc<T>);  
        } else {  
            std::sort(indexedValues.begin(), indexedValues.end());  
        }  
    
        // 提取索引  
        std::vector<int> indices;  
        for (const auto& iv : indexedValues) {  
            indices.push_back(iv.index);  
        }  
    
        return indices;  
    }

    template <class T>
    std::vector<T> linespace(T start, T stop, int num)
    {
        std::vector<T> ret;
        T step = (stop - start) / std::max(1, num - 1);
        for(int i=0;i<num;i++)
        {
            ret.push_back(i*step + start);
        }
        return ret;
    }

    template <class T>
    class Array
    {
    public:
        Array(std::vector<int> dims, bool create=false): dims_(dims)
        {
            if(create)
            {
                long sz = 1;
                for (int&d: dims) sz *= d;
                data_ = new T[sz];
            }
            updateSize();
        }

        Array(std::vector<int> dims, T* data)
        : dims_(dims)
        , data_(data)
        {
            updateSize();
        }

        void setData(T* data)
        {
            if (data_ != nullptr)
            {
                delete data_;
                // data_ = nullptr;
                data_ = data;
            }
        }

        std::vector<int> shape()
        {
            std::vector<int> ret;
            for(int i=parent_idx_.size();i<dims_.size();i++)
            {
                ret.push_back(dims_[i]);
            }
            return ret;
        }

        Array<T> ascontiguousarray()
        {
            if (parent_idx_.size())
            {
                // do something
            }
            else return *this;
        }

        T at(std::vector<int> location)
        {
            if (psz_ + location.size() == dsz_)
            {
                int idx = 0;
                // int kk = 1;
                
                for (int i=0;i<dsz_;i++)
                {
                    if (i)
                    {
                        idx *= dims_[i];
                    }

                    if (i < psz_)
                    {
                        idx += parent_idx_[i];
                    }
                    else
                    {
                        idx += location[i-psz_];
                    }
                }
                return data_[idx];
            }
            else
            {
                std::cerr << "[E] should be with shape " << shape().size() 
                          << "but got " << location.size() << std::endl;
                exit(-1);
            }
        }

        Array<T> subArray(std::vector<int> location)
        {
            Array<T> ret = Array<T>(dims_, data_);
            std::vector<int> idx = parent_idx_;
            idx.insert(idx.end(), location.begin(), location.end());
            ret.__setParentIdx(idx);
            return ret;
        }
        
        //  location[dim] = -1
        int argmaxAt(int dim, std::vector<int> location)
        {
            int ret = 0;
            location[dim] = ret;
            T maxValue = at(location);
            T currentValue = maxValue;
            for (int i=1;i<dims_[dim];i++)
            {
                location[dim] = i;
                currentValue = at(location);
                if (maxValue < currentValue)
                {
                    ret = i;
                    maxValue = currentValue;
                }
            }
            return ret;
        }

        int argminAt(int dim, std::vector<int> location)
        {
            int ret = 0;
            location[dim] = ret;
            T minValue = at(location);
            T currentValue = minValue;
            for (int i=1;i<dims_[dim];i++)
            {
                location[dim] = i;
                currentValue = at(location);
                if (minValue > currentValue)
                {
                    ret = i;
                    minValue = currentValue;
                }
            }
            return ret;
        }
        
        /**
         * do not use this function!!!!!!!
         */
        void __setParentIdx(std::vector<int> idx)
        {
            parent_idx_ = idx;
            updateSize();
        }

    private:

        void updateSize()
        {
            psz_ = parent_idx_.size();
            dsz_ = dims_.size();
        }

        int psz_=0, dsz_=0;

        std::vector<int> dims_, parent_idx_={};
        T* data_=nullptr;
    };
}

#endif
