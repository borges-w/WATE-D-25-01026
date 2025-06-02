# 加载必要包（新增tidyr）
if (!require("readxl")) install.packages("readxl")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("dplyr")) install.packages("dplyr")
if (!require("tidyr")) install.packages("tidyr")  # 新增tidyr包
library(readxl)
library(ggplot2)
library(dplyr)
library(tidyr)  # 加载tidyr

# 读取Excel数据（修改文件路径）
file_path <- "C:/Users/lenovo/Desktop/DA_R/DA.xlsx"
data <- read_excel(file_path) %>% 
  select(4, 5) %>%
  setNames(c("pH", "SSC")) %>%
  na.omit()

# 使用IQR方法清洗异常值
clean_data <- data %>%
  filter(
    pH >= quantile(pH, 0.25) - 1.5*IQR(pH) & 
      pH <= quantile(pH, 0.75) + 1.5*IQR(pH),
    SSC >= quantile(SSC, 0.25) - 1.5*IQR(SSC) & 
      SSC <= quantile(SSC, 0.75) + 1.5*IQR(SSC)
  )

# 修正后的可视化函数（使用tidyr::pivot_longer）
plot_comparison <- function(original, cleaned) {
  bind_rows(
    original %>% mutate(Data = "Original"),
    cleaned %>% mutate(Data = "Cleaned")
  ) %>%
    tidyr::pivot_longer(  # 显式调用tidyr中的函数
      cols = c(pH, SSC),
      names_to = "Variable",
      values_to = "Value"
    ) %>%
    ggplot(aes(x = Variable, y = Value, fill = Data)) +
    geom_boxplot(position = position_dodge(width = 0.8)) +
    labs(title = "Data Distribution: Before vs After IQR Outlier Removal",
         x = "Variable", y = "Value") +
    theme_minimal() +
    scale_fill_brewer(palette = "Set1")  # 使用更清晰的配色
}

# 显示对比图
plot_comparison(data, clean_data)



#第二次
library(tidyr)
library(ggplot2)
library(dplyr)

# 修正后的上下限计算函数
calculate_limits <- function(original, cleaned) {
  bind_rows(
    original %>% mutate(Data = "Original"),
    cleaned %>% mutate(Data = "Cleaned")
  ) %>%
    pivot_longer(
      cols = c(pH, SSC),
      names_to = "Variable",
      values_to = "Value"
    ) %>%
    group_by(Data, Variable) %>%
    summarise(
      Lower = quantile(Value, 0.25) - 1.5*IQR(Value),
      Upper = quantile(Value, 0.75) + 1.5*IQR(Value),
      .groups = "drop"
    )
}

# 更新后的可视化函数
plot_comparison <- function(original, cleaned) {
  # 生成合并数据
  combined <- bind_rows(
    original %>% mutate(Data = "Original"),
    cleaned %>% mutate(Data = "Cleaned")
  ) %>%
    pivot_longer(
      cols = c(pH, SSC),
      names_to = "Variable",
      values_to = "Value"
    )
  
  # 计算上下限
  limits_data <- calculate_limits(original, cleaned)
  
  # 绘图主体
  ggplot(combined, aes(x = Variable, y = Value, fill = Data)) +
    geom_boxplot(
      position = position_dodge(width = 0.8),
      outlier.shape = NA
    ) +
    # 添加参考线
    geom_segment(
      data = limits_data,
      aes(
        x = as.numeric(factor(Variable)) - 0.3,
        xend = as.numeric(factor(Variable)) + 0.3,
        y = Lower, yend = Lower
      ),
      color = "black", 
      linetype = "dashed", 
      linewidth = 0.8
    ) +
    geom_segment(
      data = limits_data,
      aes(
        x = as.numeric(factor(Variable)) - 0.3,
        xend = as.numeric(factor(Variable)) + 0.3,
        y = Upper, yend = Upper
      ),
      color = "black",
      linetype = "dashed",
      linewidth = 0.8
    ) +
    # 添加数值标签
    geom_text(
      data = limits_data,
      aes(
        x = Variable,
        y = Lower,
        label = sprintf("%.2f", Lower)
      ),
      position = position_dodge(width = 0.8),
      vjust = 1.5,
      size = 3
    ) +
    geom_text(
      data = limits_data,
      aes(
        x = Variable,
        y = Upper,
        label = sprintf("%.2f", Upper)
      ),
      position = position_dodge(width = 0.8),
      vjust = -1,
      size = 3
    ) +
    labs(x = "Variable", y = "Value") +
    theme_minimal() +
    theme(
      legend.position = "top",
      panel.grid.major.x = element_blank(),
      text = element_text(family = "sans")
    ) +
    scale_fill_manual(values = c("Original" = "#F8766D", "Cleaned" = "#00BFC4"))
}

# 执行绘图
plot_comparison(data, clean_data)



#第三次
library(ggplot2)
library(dplyr)
library(tidyr)

# 生成示例数据（替换为你的实际数据读取代码）
# data <- read_excel("C:/Users/lenovo/Desktop/DA_R/DA.xlsx") %>% select(4,5) %>% setNames(c("pH","SSC"))


  
  # IQR异常值清洗
  clean_data <- data %>%
    filter(
      between(pH, 
              quantile(pH, 0.25) - 1.5*IQR(pH),
              quantile(pH, 0.75) + 1.5*IQR(pH)),
      between(SSC,
              quantile(SSC, 0.25) - 1.5*IQR(SSC),
              quantile(SSC, 0.75) + 1.5*IQR(SSC))
    )
  
  # 计算工字线位置
  whisker_limits <- bind_rows(
    data %>% mutate(Group = "Original"),
    clean_data %>% mutate(Group = "Cleaned")
  ) %>%
    pivot_longer(
      cols = c(pH, SSC),
      names_to = "Variable",
      values_to = "Value"
    ) %>%
    group_by(Group, Variable) %>%
    summarise(
      ymin = quantile(Value, 0.25) - 1.5*IQR(Value),
      lower = quantile(Value, 0.25),
      middle = median(Value),
      upper = quantile(Value, 0.75),
      ymax = quantile(Value, 0.75) + 1.5*IQR(Value),
      .groups = "drop"
    )
  
  # 增强版箱线图绘制
  ggplot(whisker_limits, aes(x = Variable)) +
    # 绘制箱体
    geom_boxplot(
      aes(
        ymin = ymin,
        lower = lower,
        middle = middle,
        upper = upper,
        ymax = ymax,
        fill = Group
      ),
      stat = "identity",
      position = position_dodge(width = 0.8),
      width = 0.6,
      outlier.shape = NA
    ) +
    # 添加工字线
    geom_segment(
      aes(
        x = as.numeric(factor(Variable)) - 0.3,
        xend = as.numeric(factor(Variable)) + 0.3,
        y = ymin,
        yend = ymin,
        group = Group
      ),
      color = "black",
      linewidth = 0.2
    ) +
    geom_segment(
      aes(
        x = as.numeric(factor(Variable)) - 0.3,
        xend = as.numeric(factor(Variable)) + 0.3,
        y = ymax,
        yend = ymax,
        group = Group
      ),
      color = "black",
      linewidth = 0.2
    ) +
    
    scale_fill_manual(values = c("Original" = "#E69F00", "Cleaned" = "#56B4E9"))

  

  
  # 加载必要包（新增tidyr）
  if (!require("readxl")) install.packages("readxl")
  if (!require("ggplot2")) install.packages("ggplot2")
  if (!require("dplyr")) install.packages("dplyr")
  if (!require("tidyr")) install.packages("tidyr")  # 新增tidyr包
  library(readxl)
  library(ggplot2)
  library(dplyr)
  library(tidyr)  # 加载tidyr
  
  # 读取Excel数据（修改文件路径）
  file_path <- "C:/Users/lenovo/Desktop/DA_R/DA.xlsx"
  data <- read_excel(file_path) %>% 
    select(4, 5) %>%
    setNames(c("pH", "SSC")) %>%
    na.omit()
  
  # 使用IQR方法清洗异常值
  clean_data <- data %>%
    filter(
      pH >= quantile(pH, 0.25) - 1.5*IQR(pH) & 
        pH <= quantile(pH, 0.75) + 1.5*IQR(pH),
      SSC >= quantile(SSC, 0.25) - 1.5*IQR(SSC) & 
        SSC <= quantile(SSC, 0.75) + 1.5*IQR(SSC)
    )
  
  # 修正后的可视化函数（使用tidyr::pivot_longer）
  plot_comparison <- function(original, cleaned) {
    bind_rows(
      original %>% mutate(Data = "Original"),
      cleaned %>% mutate(Data = "Cleaned")
    ) %>%
      tidyr::pivot_longer(  # 显式调用tidyr中的函数
        cols = c(pH, SSC),
        names_to = "Variable",
        values_to = "Value"
      ) %>%
      ggplot(aes(x = Variable, y = Value, fill = Data)) +
      geom_boxplot(position = position_dodge(width = 0.8)) +
      labs(title = "Data Distribution: Before vs After IQR Outlier Removal",
           x = "Variable", y = "Value") +
      theme_minimal() +
      scale_fill_brewer(palette = "Set1")  # 使用更清晰的配色
  }
  
  # 显示对比图
  plot_comparison(data, clean_data)
  

  
####可用但缺少上下限  
# 加载必要包（新增tidyr）
  if (!require("readxl")) install.packages("readxl")
  if (!require("ggplot2")) install.packages("ggplot2")
  if (!require("dplyr")) install.packages("dplyr")
  if (!require("tidyr")) install.packages("tidyr")  # 新增tidyr包
  library(readxl)
  library(ggplot2)
  library(dplyr)
  library(tidyr)  # 加载tidyr
  
  # 读取Excel数据（修改文件路径）
  file_path <- "C:/Users/lenovo/Desktop/DA_R/DA.xlsx"
  data <- read_excel(file_path) %>% 
    select(4, 5) %>%
    setNames(c("pH", "SSC")) %>%
    na.omit()
  
  # 使用IQR方法清洗异常值
  clean_data <- data %>%
    filter(
      pH >= quantile(pH, 0.25) - 1.5*IQR(pH) & 
        pH <= quantile(pH, 0.75) + 1.5*IQR(pH),
      SSC >= quantile(SSC, 0.25) - 1.5*IQR(SSC) & 
        SSC <= quantile(SSC, 0.75) + 1.5*IQR(SSC)
    )
  
  # 修正后的可视化函数（使用tidyr::pivot_longer）
  plot_comparison <- function(original, cleaned) {
    bind_rows(
      original %>% mutate(Data = "Original"),
      cleaned %>% mutate(Data = "Cleaned")
    ) %>%
      tidyr::pivot_longer(  # 显式调用tidyr中的函数
        cols = c(pH, SSC),
        names_to = "Variable",
        values_to = "Value"
      ) %>%
      ggplot(aes(x = Variable, y = Value, fill = Data)) +
      geom_boxplot(position = position_dodge(width = 0.8)) +
      labs(title = "Data Distribution: Before vs After IQR Outlier Removal",
           x = "Variable", y = "Value") +
      theme_minimal() +
      scale_fill_brewer(palette = "Set1")  # 使用更清晰的配色
  }
  
  # 显示对比图
  plot_comparison(data, clean_data)
  
  
  
  
  
  # 加载必要包
  if (!require("readxl")) install.packages("readxl")
  if (!require("ggplot2")) install.packages("ggplot2")
  if (!require("dplyr")) install.packages("dplyr")
  if (!require("tidyr")) install.packages("tidyr")
  library(readxl)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  
  # 读取Excel数据（修改文件路径）
  file_path <- "C:/Users/lenovo/Desktop/DA_R/DA.xlsx"
  data <- read_excel(file_path) %>% 
    select(4, 5) %>%
    setNames(c("pH", "SSC")) %>%
    na.omit()
  
  # 使用IQR方法清洗异常值
  clean_data <- data %>%
    filter(
      pH >= quantile(pH, 0.25) - 1.5*IQR(pH) & 
        pH <= quantile(pH, 0.75) + 1.5*IQR(pH),
      SSC >= quantile(SSC, 0.25) - 1.5*IQR(SSC) & 
        SSC <= quantile(SSC, 0.75) + 1.5*IQR(SSC)
    )
  
  # 修正后的可视化函数
  plot_comparison <- function(original, cleaned) {
    # 合并数据
    combined <- bind_rows(
      original %>% mutate(Data = "Original"),
      cleaned %>% mutate(Data = "Cleaned")
    ) %>%
      tidyr::pivot_longer(
        cols = c(pH, SSC),
        names_to = "Variable",
        values_to = "Value"
      )
    
    # 计算箱线图实际位置
    plot_data <- ggplot_build(
      ggplot(combined, aes(Variable, Value, fill = Data)) +
        geom_boxplot(position = position_dodge(width = 0.8))
    )$data[[1]] %>%
      as_tibble() %>%
      select(group, x, xmin, xmax, ymin, ymax) %>%
      mutate(
        Variable = levels(factor(combined$Variable))[ceiling(group/2)],
        Data = rep(c("Original", "Cleaned"), times = n()/2)
      )
    
    # 添加工字线图层
    ggplot(combined, aes(x = Variable, y = Value, fill = Data)) +
      geom_boxplot(position = position_dodge(width = 0.8)) +
      # 下边缘线
      geom_segment(
        data = plot_data,
        aes(x = xmin, xend = xmax, y = ymin, yend = ymin),
        linewidth = 0.8, color = "black"
      ) +
      # 上边缘线
      geom_segment(
        data = plot_data,
        aes(x = xmin, xend = xmax, y = ymax, yend = ymax),
        linewidth = 0.8, color = "black"
      ) +
      labs(x = "Variable", y = "Value") +
      theme_minimal() +
      scale_fill_brewer(palette = "Set1")
  }
  
  # 显示对比图
  plot_comparison(data, clean_data)

  
  
  
  
  ##chat8
  # 加载必要包（新增tidyr）
  if (!require("readxl")) install.packages("readxl")
  if (!require("ggplot2")) install.packages("ggplot2")
  if (!require("dplyr")) install.packages("dplyr")
  if (!require("tidyr")) install.packages("tidyr")  # 新增tidyr包
  library(readxl)
  library(ggplot2)
  library(dplyr)
  library(tidyr)  # 加载tidyr
  
  # 读取Excel数据（修改文件路径）
  file_path <- "C:/Users/lenovo/Desktop/DA_R/DA.xlsx"
  data <- read_excel(file_path) %>% 
    select(4, 5) %>%
    setNames(c("pH", "SSC")) %>%
    na.omit()
  
  # 使用IQR方法清洗异常值
  clean_data <- data %>%
    filter(
      pH >= quantile(pH, 0.25) - 1.5*IQR(pH) & 
        pH <= quantile(pH, 0.75) + 1.5*IQR(pH),
      SSC >= quantile(SSC, 0.25) - 1.5*IQR(SSC) & 
        SSC <= quantile(SSC, 0.75) + 1.5*IQR(SSC)
    )
  
  # 修正后的可视化函数（使用tidyr::pivot_longer）
  plot_comparison <- function(original, cleaned) {
    bind_rows(
      original %>% mutate(Data = "Original"),
      cleaned %>% mutate(Data = "Cleaned")
    ) %>%
      tidyr::pivot_longer(  # 显式调用tidyr中的函数
        cols = c(pH, SSC),
        names_to = "Variable",
        values_to = "Value"
      ) %>%
      ggplot(aes(x = Variable, y = Value, fill = Data)) +
      geom_boxplot(position = position_dodge(width = 0.8)) +
      
      # 计算上下限并添加横线
      stat_boxplot(geom = "errorbar", width = 0.2, position = position_dodge(width = 0.8)) +
      stat_summary(fun.data = "mean_cl_normal", geom = "point", shape = 18, size = 3) + 
      
      # 上下限横线
      geom_segment(
        aes(x = 1, xend = 1, y = quantile(Value, 0.25) - 1.5*IQR(Value), yend = quantile(Value, 0.75) + 1.5*IQR(Value)),
        color = "black", size = 1) +  
      
      labs(title = "Data Distribution: Before vs After IQR Outlier Removal",
           x = "Variable", y = "Value") +
      theme_minimal() +
      scale_fill_brewer(palette = "Set1")  # 使用更清晰的配色
  }
  
  # 显示对比图
  plot_comparison(data, clean_data)

  
  
  #deepseek
  # 加载必要包
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  
  # 读取并处理数据（路径需自行修改）
  file_path <- "C:/Users/lenovo/Desktop/DA_R/DA.xlsx"
  data <- readxl::read_excel(file_path) %>% 
    select(4, 5) %>%
    setNames(c("pH", "SSC")) %>%
    na.omit()
  
  # IQR清洗数据
  clean_data <- data %>%
    filter(
      pH >= quantile(pH, 0.25) - 1.5*IQR(pH) & 
        pH <= quantile(pH, 0.75) + 1.5*IQR(pH),
      SSC >= quantile(SSC, 0.25) - 1.5*IQR(SSC) & 
        SSC <= quantile(SSC, 0.75) + 1.5*IQR(SSC)
    )
  
  # 计算上下限并输出
  calculate_limits <- function(df, group_name) {
    df %>%
      summarise(
        pH_lower = quantile(pH, 0.25) - 1.5*IQR(pH),
        pH_upper = quantile(pH, 0.75) + 1.5*IQR(pH),
        SSC_lower = quantile(SSC, 0.25) - 1.5*IQR(SSC),
        SSC_upper = quantile(SSC, 0.75) + 1.5*IQR(SSC)
      ) %>%
      mutate(Group = group_name) %>%
      select(Group, everything())
  }
  
  # 输出上下限表格
  limits_table <- bind_rows(
    calculate_limits(data, "Original"),
    calculate_limits(clean_data, "Cleaned")
  )
  print(limits_table)
  
  # 可视化函数
  plot_comparison <- function(original, cleaned) {
    combined <- bind_rows(
      original %>% mutate(Data = "Original"),
      cleaned %>% mutate(Data = "Cleaned")
    ) %>%
      pivot_longer(c(pH, SSC), names_to = "Variable", values_to = "Value")
    
    # 获取箱线图实际位置数据
    plot_data <- ggplot_build(
      ggplot(combined, aes(Variable, Value, fill = Data)) +
        geom_boxplot(position = position_dodge(0.8))
    )$data[[1]] %>%
      as_tibble() %>%
      mutate(
        Variable = if_else(group %% 2 == 1, "pH", "SSC"),
        Data = if_else(group <= 2, "Original", "Cleaned")
      )
    
    # 调整工字线长度（原长度的80%）
    line_ratio <- 0.4  # 调整此参数控制线长（0.5=50%箱宽）
    plot_data <- plot_data %>%
      mutate(
        x_center = (xmin + xmax)/2,
        x_start = x_center - (xmax - xmin)*line_ratio/2,
        x_end = x_center + (xmax - xmin)*line_ratio/2
      )
    
    ggplot(combined, aes(x = Variable, y = Value, fill = Data)) +
      geom_boxplot(position = position_dodge(0.8), width = 0.7) +
      # 添加工字线
      geom_segment(
        data = plot_data,
        aes(x = x_start, xend = x_end, y = ymin, yend = ymin),
        linewidth = 1, color = "black"
      ) +
      geom_segment(
        data = plot_data,
        aes(x = x_start, xend = x_end, y = ymax, yend = ymax),
        linewidth = 1, color = "black"
      ) +
      labs(x = "Variable", y = "Value") +
      theme_minimal() +
      scale_fill_brewer(palette = "Set1") +
      theme(legend.position = "top")
  }
  
  # 生成图形
  plot_comparison(data, clean_data)
  
  
  
  
  #解决出现负值
  # 加载必要包
  library(readxl)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  
  # 读取数据（路径需修改）
  file_path <- "C:/Users/lenovo/Desktop/DA_R/DA.xlsx"
  data <- read_excel(file_path) %>% 
    select(4, 5) %>%
    setNames(c("pH", "SSC")) %>%
    mutate(across(c(pH, SSC), as.numeric)) %>%  # 强制转换为数值型
    na.omit()
  
  # 检查原始数据范围
  cat("Original SSC range:", range(data$SSC), "\n")
  
  # IQR清洗数据
  clean_data <- data %>%
    filter(
      pH >= quantile(pH, 0.25) - 1.5*IQR(pH) & 
        pH <= quantile(pH, 0.75) + 1.5*IQR(pH),
      SSC >= quantile(SSC, 0.25, type = 2) - 1.5*IQR(SSC) &  # 指定分位数计算方式
        SSC <= quantile(SSC, 0.75, type = 2) + 1.5*IQR(SSC)
    )
  
  # 检查清洗后数据范围
  cat("Cleaned SSC range:", range(clean_data$SSC), "\n")
  
  # 修正后的上下限计算函数
  calculate_limits <- function(df, group_name) {
    df %>%
      summarise(
        pH_lower = pmax(quantile(pH, 0.25, type = 2) - 1.5*IQR(pH), min(pH)),
        pH_upper = pmin(quantile(pH, 0.75, type = 2) + 1.5*IQR(pH), max(pH)),
        SSC_lower = pmax(quantile(SSC, 0.25, type = 2) - 1.5*IQR(SSC), min(SSC)),
        SSC_upper = pmin(quantile(SSC, 0.75, type = 2) + 1.5*IQR(SSC), max(SSC))
      ) %>%
      mutate(Group = group_name) %>%
      select(Group, everything())
  }
  
  # 输出上下限
  limits_table <- bind_rows(
    calculate_limits(data, "Original"),
    calculate_limits(clean_data, "Cleaned")
  )
  print(limits_table)
  
  
  
#最终
  # 加载必要包
  library(readxl)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  
  # 读取数据（路径需修改）
  file_path <- "C:/Users/lenovo/Desktop/DA_R/DA.xlsx"
  data <- read_excel(file_path) %>% 
    select(4, 5) %>%
    setNames(c("pH", "SSC")) %>%
    mutate(across(c(pH, SSC), as.numeric)) %>%  # 强制数值类型
    na.omit()
  
  # 数据验证
  cat("原始数据范围:\n")
  cat("pH :", range(data$pH), "\n")
  cat("SSC:", range(data$SSC), "\n\n")
  
  # IQR清洗数据（使用type=2分位数计算）
  clean_data <- data %>%
    filter(
      pH >= pmax(quantile(pH, 0.25, type=2) - 1.5*IQR(pH), min(pH)),
      pH <= pmin(quantile(pH, 0.75, type=2) + 1.5*IQR(pH), max(pH)),
      SSC >= pmax(quantile(SSC, 0.25, type=2) - 1.5*IQR(SSC), min(SSC)),
      SSC <= pmin(quantile(SSC, 0.75, type=2) + 1.5*IQR(SSC), max(SSC))
    )
  
  # 清洗后数据验证
  cat("清洗后数据范围:\n")
  cat("pH :", range(clean_data$pH), "\n")
  cat("SSC:", range(clean_data$SSC), "\n")
  
  # 计算上下限并输出
  calculate_limits <- function(df, group_name) {
    df %>%
      summarise(
        pH_lower = pmax(quantile(pH, 0.25, type=2) - 1.5*IQR(pH), min(pH)),
        pH_upper = pmin(quantile(pH, 0.75, type=2) + 1.5*IQR(pH), max(pH)),
        SSC_lower = pmax(quantile(SSC, 0.25, type=2) - 1.5*IQR(SSC), min(SSC)),
        SSC_upper = pmin(quantile(SSC, 0.75, type=2) + 1.5*IQR(SSC), max(SSC))
      ) %>%
      mutate(Group = group_name)
  }
  
  # 生成上下限表格
  limits_table <- bind_rows(
    calculate_limits(data, "Original"),
    calculate_limits(clean_data, "Cleaned")
  )
  print(limits_table)
  
  # 可视化函数
  plot_comparison <- function(original, cleaned) {
    # 合并数据
    combined <- bind_rows(
      original %>% mutate(Data = "Original"),
      cleaned %>% mutate(Data = "Cleaned")
    ) %>%
      pivot_longer(c(pH, SSC), names_to = "Variable", values_to = "Value")
    
    # 获取箱线图渲染数据
    base_plot <- ggplot(combined, aes(Variable, Value, fill=Data)) +
      geom_boxplot(position=position_dodge(0.8), width=0.6)
    
    plot_data <- ggplot_build(base_plot)$data[[1]] %>%
      as_tibble() %>%
      mutate(
        Variable = levels(factor(combined$Variable))[ceiling(group/2)],
        Data = rep(c("Original", "Cleaned"), length.out=n())
      )
    
    # 工字线参数设置
    line_ratio <- 0.35  # 工字线长度比例
    plot_data <- plot_data %>%
      mutate(
        x_center = x,
        x_start = x_center - (0.6*line_ratio)/2,  # 0.6为箱体宽度
        x_end = x_center + (0.6*line_ratio)/2
      )
    
    # 构建最终图形
    base_plot +
      # 下边缘线
      geom_segment(
        data = plot_data,
        aes(x=x_start, xend=x_end, y=ymin, yend=ymin),
        linewidth=0.8, color="black"
      ) +
      # 上边缘线
      geom_segment(
        data = plot_data,
        aes(x=x_start, xend=x_end, y=ymax, yend=ymax),
        linewidth=0.8, color="black"
      ) +
      labs(x="Variable", y="Value") +
      theme_minimal() +
      scale_fill_brewer(palette="Set1") +
      theme(
        legend.position = "top",
        panel.grid.major.x = element_blank()
      )
  }
  
  # 生成最终图形
  plot_comparison(data, clean_data)
  