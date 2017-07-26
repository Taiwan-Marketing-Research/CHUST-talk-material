######################################################
#                                                    #
#                   鉅亨網 crawler                   #
#                                                    #
######################################################

#author: Howard Chung
#original date: 2017/3/28
#updated date: 2017/3/28 
#pupose: crawl html table of stock from 鉅亨網
#version: v 0.1.0
#update news: 
#' 2017/3/28 : crawl stock from 鉅亨網 and function of auto-crawling done

#' [鉅亨網](http://www.cnyes.com/twstock/ps_historyprice/2330.htm)

knitr::opts_knit$set(root.dir = '..')

######################## 1.library loaded ##############################
rm(list = ls())
library(magrittr) #for pipeline 
library(httr) #mainly for web crawling
library(rvest) #for html table format on unix OS
suppressPackageStartupMessages(library(XML)) #for html table format on Windows OS
library(DT) # build interactive table

######################## 2. Request: GET crawler made ##############################
url <- "http://www.cnyes.com/twstock/ps_historyprice/2498.htm" #宏達電

# make a GET request and have response
res <- GET(url = url)
#
######################## 3. parser : extract data wanted ##############################

# encode text-content into utf-8
doc_str <- res %>% 
  content(as = "text", encoding = "UTF-8")

# parse data into table format
if (.Platform$OS.type == "unix") { 
  dat <- doc_str %>% 
    read_html() %>% 
    html_table(fill = TRUE) %>% 
    .[[2]] # extract the second component of the html table
} else if (.Platform$OS.type == "windows") { 
  # because html_table() doesn't work on Windows T^T
  
  #get stockname: for list names used
  stockname <- doc_str %>% 
    read_html(encoding = "UTF-8") %>%
    as.character() %>% 
    #gsub is kind of regular express(regex) for extracting string from unstructured texts
    #.*? 【insert the patterned text before string you wanna extract】 (.*?) 【insertthe patterned text after string you wanna extract】
    gsub(pattern = ".*?\n\t(.*?)_.*","\\1",.)
  
  # get and parse data
  dat <- doc_str %>% 
    read_html(encoding = "UTF-8") %>%
    as.character() %>%
    XML::readHTMLTable(encoding = "UTF-8") %>% 
    .[[2]] %>% # extract the second component of the html table
    `colnames<-`(c("日期", "開盤", "最高", "最低", "收盤",
                   "漲跌", "漲%", "成交量", "成交金額", "本益比")) # set column names
}

######################## 4.parser : clean and organize data ############################
str(dat) # see the structure of dat
head(dat)

# convert data type from character to numeric variable
dat[, 2:ncol(dat)] <- sapply(dat[, 2:ncol(dat)], 
                             function(x){
                               x <- gsub(",|%", "", x) # replace , and % with ""
                               as.numeric(x)
                             })

# convert data type from character to date
dat$日期 <- as.Date(dat$日期)

# add names of selected stock to the data frame
dat$股票名稱=stockname

# subset column
dat=subset(x = dat, select=c("股票名稱","日期", "開盤", "最高", "最低", "收盤",
                            "漲跌", "漲%", "成交量", "成交金額", "本益比"))


str(dat)
head(dat, 10)


#Add error handlers
#error handlers: first add 【trycatch, status code】
get_stock_price <- function(stock_id) {
  url <- sprintf("http://www.cnyes.com/twstock/ps_historyprice/%i.htm", stock_id)
  tryCatch({
    # make a GET request
    res <- GET(url = url)
    # error detect by status code
    if (res[["status_code"]] >=400) { 
      # error handle
      message(sprintf("Stock %i has error of %i", stock_id, res[["status_code"]]))
      return(NULL)
    }
    
    doc_str <- res %>% 
      content(as = "text", encoding = "UTF-8")
    
    if (.Platform$OS.type == "unix") { 
      # get and parse data
      dat <- doc_str %>% 
        read_html() %>% 
        html_table(fill = TRUE) %>% 
        .[[2]] # extract the second component of the html table
    } else if (.Platform$OS.type == "windows") { # html_table() doesn't work on Windows
      
      #get stockname
      stockname <- doc_str %>% 
        read_html(encoding = "UTF-8") %>%
        as.character()  
      
      # get and parse data
      dat <- doc_str %>% 
        read_html(encoding = "UTF-8") %>%
        as.character() %>%
        XML::readHTMLTable(encoding = "UTF-8") %>% 
        .[[2]] %>% # extract the second component of the html table
        `colnames<-`(c("日期", "開盤", "最高", "最低", "收盤",
                       "漲跌", "漲%", "成交量", "成交金額", "本益比")) # set column names
    }
    
    # return NULL if no data was retrieved
    if (nrow(dat) == 0) {
      message(sprintf("Stock %s has no data.", stock_id))
      return(NULL)
    }
    
    # convert data type from character to double
    dat[, 2:ncol(dat)] <- sapply(dat[, 2:ncol(dat)], 
                                 function(x){
                                   x <- gsub(",|%", "", x)
                                   as.double(x)
                                 })
    
    # convert data type from character to date
    dat$`日期` %<>% as.Date()
    
    #add names of selected stock to the data frame
    dat$股票名稱=gsub(pattern = ".*?\n\t(.*?)_.*","\\1",stockname)
    
    #subset col
    dat=subset(x = dat,select=c("股票名稱","日期", "開盤", "最高", "最低", "收盤",
                                "漲跌", "漲%", "成交量", "成交金額", "本益比"))
    
    
    return(dat)
  }, error = function(cond) {
    # return NULL if the error "subscript out of bounds" happens
    if (cond$call == ".[[2]]") { 
      message(sprintf("Stock %s has no data.", stock_id))
      return(NULL)
    }
  })
}

######################## 5.function : Retrieve Data through the function ##############################
dat <- get_stock_price(stock_id = 3450)
DT::datatable(dat) #interactive data table

######################## 6.function : Retrieve Meta-Data through loop  ##############################
dataa=list()
for (i in 2330:2500){
  name=paste("stock No.", i, sep="")
  dataa[name]=list(get_stock_price(stock_id = i))
}

