#pragma once
#ifndef PYLIKE_YAML_UTILS_H
#define PYLIKE_YAML_UTILS_H

#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>
#include "./str.h"
#include "./logger.h"
#include "./tabulate.h"

namespace YAML {

template <class T>
static void param_env(YAML::Node& cfg, std::string key, T& param_, T default_)
{

    if (cfg[key].IsDefined())
    {
        pystring str = cfg[key].as<std::string>();
        if (str.startswith("$"))
        {
            pystring env_name = str.split("$")[1].split({",", " ", ":", ";", ".", "/", "[", "]", "'", "|", "-", "+", "(", ")", "!", "@", "#", "%", "^"})[0];
            const char *result = std::getenv(env_name.c_str());
            if (result != nullptr)
            {
                str = std::string(result);
                param_ = (T)str;
            }
            else
            {
                param_ = default_;
                WARN << "env " << env_name << " not found, use " << default_ << " as default." << ENDL;
            }
        }
        else
        {
            param_ = cfg[key].as<T>();
        }
    }
    else
    {
        param_ = default_;
    }
}

template <class T>
static void param(YAML::Node& cfg, std::string key, T& param_, T default_)
{
    if (cfg[key].IsDefined())
    {
        param_ = cfg[key].as<T>();
    }
    else
    {
        param_ = default_;
    }
}

// 暂时不支持列表中有字典
static std::string __parse_scalar(YAML::Node config)
{
    std::stringstream oss;
    oss << "[";
    for (int i=0;i<config.size();++i)
    {
        if(i) oss << ", ";
        if (config[i].IsSequence()) oss << __parse_scalar(config[i]);
        else oss << config[i].as<std::string>();
    }
    oss << "]";
    return oss.str();
}

static std::string show(YAML::Node& config, tabulate::TableStyle style=TS_FANCY_OUTLINE, bool show_inside=false)
{
    std::vector<pystring> subConfigsKey;
    tabulate::Table t;
    // t.clear();
    t.setDefaultStyle(style);
    t.setAlign({TA_LEFT, TA_LEFT});
    t.setHead() << "Keys" << "Values" << t.endLine();
    for (const auto& key_value : config) {
        std::string key = key_value.first.as<std::string>();
        YAML::Node value = key_value.second;

        // 如果值是一个映射（即另一个 YAML 对象），则递归打印
        if (value.IsMap()) 
        {
            t.setContent() << key << "<dict>, see 'Table for " + key + "'" << t.endLine();
            pystring this_string = "\n Table for " + key + "\n" + show(value, style, false);
            subConfigsKey.push_back(this_string.replace("\n", "\n    "));
        }
        else if (value.IsScalar()) {
            // 如果值是标量（如字符串、整数等），则直接打印值
            // std::cout << "  Value: " << value.as<std::string>() << std::endl;
            t.setContent() << key << value.as<std::string>() << t.endLine();
        }
        // 可以添加更多类型检查和处理逻辑，如处理序列（数组）等
        else if (value.IsSequence())
        {
            t.setContent() << key << __parse_scalar(value) << t.endLine();
        }
    }
    std::string ret = t.toString();
    for (auto& str: subConfigsKey)
    {
        ret += "\n" + str.str();
    }
    if (show_inside)
    {
        std::cout << ret << std::endl;
    }
    return ret;
}


}

#endif
