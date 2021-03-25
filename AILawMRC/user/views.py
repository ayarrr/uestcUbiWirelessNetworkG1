import re
import datetime
import time

from django.shortcuts import render
from django.shortcuts import HttpResponse
from django.http.response import JsonResponse
from user import models

crawlSp = []
crawlFp = []
crawlTotal = []
crawlId = []
needcrawl = []

# Create your views here.
def register(request):
    print('进入接口register')
    json = {}
    if request.method == 'POST':
        name = request.POST.get('name').strip()
        phone = request.POST.get('phone').strip()
        email = request.POST.get('email').strip()
        password = request.POST.get('password').strip()
        checkpwd = request.POST.get('checkpwd').strip()

        if name == '' or phone == '' or email == '' or password == '' or checkpwd == '':
            json['resultCode'] = '20001'
            json['resultDesc'] = '参数不全'
        else:
            try:
                user = models.User.objects.get(u_phone=phone)
                json['resultCode'] = '10002'
                json['resultDesc'] = '该手机号已被注册'
            except:
                try:
                    user = models.User.objects.get(u_email=email)
                    json['resultCode'] = '10003'
                    json['resultDesc'] = '该邮箱已被注册'
                except:
                    phoneRe = re.match(r'^1[35678]\d{9}$', phone)
                    emailRe = re.match(r'^[0-9a-zA-Z_]{0,19}@[0-9a-zA-Z]{1,13}\.[com,cn,net]{1,3}$', email)
                    if phoneRe and emailRe and len(password) >= 6:
                        if password == checkpwd:
                            try:
                                user = models.User.objects.create(u_name=name, u_pwd=password, u_phone=phone, u_email=email)
                                json['resultCode'] = '10001'
                                json['resultDesc'] = '注册成功'
                            except:
                                json['resultCode'] = '30000'
                                json['resultDesc'] = '服务器故障'
                        else:
                            json['resultCode'] = '20002'
                            json['resultDesc'] = '两次密码不一致'
                    else:
                        json['resultCode'] = '10005'
                        json['resultDesc'] = '参数格式错误'
    return JsonResponse(json)

def login(request):
    print('进入接口login')
    json = {}
    if request.method == "POST":
        username = request.POST.get("username").strip()
        password = request.POST.get("password").strip()

        if username == '' or password == '':
            json['resultCode'] = '20000'
            json['resultDesc'] = '参数不全'
        else:
            phoneFlag = True
            emailFlag = True
            try:
                user = models.User.objects.get(u_phone=username)
            except:
                phoneFlag = False

            if phoneFlag:
                if user.u_pwd == password:
                    json['resultCode'] = '10001'
                    json['resultDesc'] = '登陆成功'
                    request.session['u_id'] = user.u_id
                else:
                    json['resultCode'] = '10003'
                    json['resultDesc'] = '密码错误'
            else:
                try:
                    user = models.User.objects.get(u_email=username)
                except:
                    emailFlag = False

                if emailFlag:
                    print("邮箱存在")
                    if user.u_pwd == password:
                        json['resultCode'] = '10001'
                        json['resultDesc'] = '登陆成功'
                        request.session['u_id'] = user.u_id
                        print(request.session.get('u_id'))
                        print(request.session['u_id'])
                    else:
                        json['resultCode'] = '10003'
                        json['resultDesc'] = '密码错误'
                else:
                    json['resultCode'] = '10002'
                    json['resultDesc'] = '用户不存在'

    return JsonResponse(json)

def updatePwd(request):
    print('进入接口updatePwd')
    json = {}
    if request.method == "POST":
        oldpwd = request.POST.get("oldpwd").strip()
        newpwd = request.POST.get("newpwd").strip()
        checkpwd = request.POST.get("checkpwd").strip()

        if oldpwd == '' or newpwd == '' or checkpwd == '':
            json['resultCode'] = '20000'
            json['resultDesc'] = '参数不全'
        else:
            u_id = 1
            # u_id = request.session.get('u_id')
            try:
                user = models.User.objects.get(u_id=u_id)
                if user.u_pwd == oldpwd:
                    if len(newpwd) >= 6:
                        if newpwd == checkpwd:
                            user.u_pwd = newpwd
                            user.save()
                            json['resultCode'] = '10001'
                            json['resultDesc'] = '修改成功'
                        else:
                            json['resultCode'] = '10007'
                            json['resultDesc'] = '两次新密码不一致'
                    else:
                        json['resultCode'] = '10006'
                        json['resultDesc'] = '新密码格式错误'
                else:
                    json['resultCode'] = '10005'
                    json['resultDesc'] = '原密码错误'
            except:
                json['resultCode'] = '30000'
                json['resultDesc'] = '服务器故障'

    return JsonResponse(json)

def updateInfo(request):
    print('进入接口updateInfo')
    json = {}
    if request.method == "POST":
        name = request.POST.get("name").strip()
        phone = request.POST.get("phone").strip()
        email = request.POST.get("email").strip()

        if name == '' or phone == '' or email == '':
            json['resultCode'] = '20000'
            json['resultDesc'] = '参数不全'
        else:
            u_id = 1
            # u_id = request.session.get('u_id')
            json['resultCode'] = '10001'
            json['resultDesc'] = '修改成功'
            try:
                user = models.User.objects.get(u_id=u_id)

                phoneNotUsed = False
                emailNotUsed = False

                if user.u_phone != phone:
                    try:
                        other_user = models.User.objects.get(u_phone=phone)
                        json['resultCode'] = '10002'
                        json['resultDesc'] = '新手机号已被注册'
                    except:
                        phoneNotUsed = True

                if user.u_email != email:
                    try:
                        other_user = models.User.objects.get(u_email=email)
                        json['resultCode'] = '10003'
                        json['resultDesc'] = '新邮箱已被注册'
                    except:
                        emailNotUsed = True

                if phoneNotUsed or emailNotUsed:
                    phoneRe = re.match(r'^1[35678]\d{9}$', phone)
                    emailRe = re.match(r'^[0-9a-zA-Z_]{0,19}@[0-9a-zA-Z]{1,13}\.[com,cn,net]{1,3}$', email)
                    if not phoneRe or not emailRe:
                        json['resultCode'] = '10005'
                        json['resultDesc'] = '手机号或邮箱格式错误'

            except:
                json['resultCode'] = '30000'
                json['resultDesc'] = '服务器故障'

            if json['resultCode'] == '10001':
                user.u_name = name
                user.u_phone = phone
                user.u_email = email
                user.save()

    return JsonResponse(json)

def crawl(request):
    print('进入接口crawl')
    json = {}
    if request.method == "POST":
        keyword = request.POST.get("keyword")
        pagenum = request.POST.get("pagenum")
        #标识是否为重新爬取
        crawl_flag = request.POST.get("flag")
        #若为重新爬取则发送原爬取记录主键
        crawl_id = int(request.POST.get("cr_id"))

        if keyword.strip() == '' or pagenum.strip() == '':
            json['resultCode'] = '20000'
            json['resultDesc'] = '参数不全'
        else:
            pagenum = int(pagenum)
            u_id = 1
            # u_id = request.session.get('u_id')
            #爬取时间
            cr_time_normal = datetime.datetime.now()
            cr_time_str = str(cr_time_normal).split('.')[0]
            cr_time = datetime.datetime.strptime(cr_time_str, "%Y-%m-%d %H:%M:%S")
            print(cr_time)

            #创建记录
            try:
                #若为失败任务重新爬取则先删除原记录
                if crawl_flag == 'again':
                    models.Ciconnect.objects.filter(cr_id=crawl_id).delete()
                    models.Crawlrecord.objects.get(cr_id=crawl_id).delete()
                now_cr = models.Crawlrecord.objects.create(cr_keyword=keyword, cr_num=pagenum, cr_time=cr_time, u_id=u_id, cr_status=0, cr_check=0)
                now_crid = now_cr.cr_id
                print(now_crid)

                try:
                    # 相同用户重复查询相同关键词及篇章，直接根据之前记录得到相应文书，添加到这次记录中
                    pre_cr = models.Crawlrecord.objects.filter(cr_keyword=keyword, cr_num=pagenum, u_id=u_id, cr_status=1).first()
                    pre_crid = pre_cr.cr_id
                    instrumentids = models.Ciconnect.objects.filter(cr_id=pre_crid)

                    # 添加对应文书记录
                    try:
                        for i in instrumentids:
                            ci = models.Ciconnect.objects.create(cr_id=now_crid, i_id=i.i_id)
                        json['resultCode'] = '10001'
                        json['resultDesc'] = '操作成功'
                    # 出错
                    except:
                        json['resultCode'] = '30000'
                        json['resultDesc'] = '服务器故障'
                except:
                    # 若不是重复查询相同关键词及篇章
                    # 首先确认该关键词在数据库中是否存在对应文书
                    print('逻辑正确')
                    instrument_list = models.Instrument.objects.filter(cr_keyword=keyword)
                    instrument_len = instrument_list.count()
                    print(instrument_len)

                    # 爬虫起点
                    startindex = -1
                    # 爬虫篇数
                    crawlnum = 0

                    # 若该关键词下不存在文书，则从头开始爬取相应篇数
                    print('ok')
                    if instrument_len <= 0:
                        startindex = 0
                        crawlnum = pagenum
                    # 若该关键词下存在文书，确认数据库中文书数量是否满足查询数量
                    else:
                        # 如果数据库文书不够
                        if instrument_len < pagenum:
                            startindex = instrument_len
                            crawlnum = pagenum - instrument_len
                            #将已有文书先全部添加与当前记录的对应关系
                            for i in instrument_list:
                                ci = models.Ciconnect.objects.create(cr_id=now_crid, i_id=i.i_id)
                        # 如果数据库文书足够，从已有文书中选取pagenum篇，生成相应记录
                        else:
                            get_instrument = instrument_list[:pagenum]
                            try:
                                for i in get_instrument:
                                    ci = models.Ciconnect.objects.create(cr_id=now_crid, i_id=i.i_id)
                                json['resultCode'] = '10001'
                                json['resultDesc'] = '操作成功'
                            except:
                                json['resultCode'] = '30000'
                                json['resultDesc'] = '服务器故障'

                    crawlId.append(now_crid)
                    list_index = crawlId.index(now_crid)
                    print(list_index)
                    # 如果不需要爬虫
                    if startindex == -1:
                        needcrawl.append(False)
                    else:
                        print('ok')
                        needcrawl.append(True)
                        # 进行爬虫
                        from selenium import webdriver
                        from selenium.webdriver.chrome.options import Options
                        import time
                        import hashlib

                        # #正常模式
                        # chrome_options = webdriver.ChromeOptions()
                        # chrome_options.add_argument('--start-maximized')
                        # #无头模式启动
                        # chrome_options.add_argument('--headless')
                        # #谷歌文档提到需要加上这个属性来规避bug
                        # chrome_options.add_argument('--disable-gpu')
                        # plugin_file = './spider/utils/proxy_auth_plugin.zip'
                        # chrome_options.add_extension(plugin_file)

                        # firefox-headless
                        # from selenium import webdriver
                        # options = webdriver.FirefoxOptions()
                        # options.set_headless()
                        # # options.add_argument('-headless')
                        # options.add_argument('--disable-gpu')
                        # driver = webdriver.Firefox(firefox_options=options)
                        # driver.get('http://httpbin.org/user-agent')
                        # driver.get_screenshot_as_file('test.png')
                        # driver.close()

                        # chrome-headless
                        chrome_options = Options()
                        # 无头模式启动
                        chrome_options.add_argument('--headless')
                        # 谷歌文档提到需要加上这个属性来规避bug
                        chrome_options.add_argument('--disable-gpu')
                        chrome_options.add_argument('--start-maximized')
                        # 初始化实例
                        driver = webdriver.Chrome(options=chrome_options)
                        #    self.browser = webdriver.Chrome(chrome_options=self.chrome_options)
                        # wait = WebDriverWait(driver, TIMEOUT)
                        urlnew = "https://www.itslaw.com/search?searchMode=judgements&sortType=1&conditions=searchWord%2B%E6%B3%95%E5%BE%8B%2B1%2B%E6%B3%95%E5%BE%8B&searchView=text"
                        url = "https://wusong.itslaw.com/search?searchMode=judgements&sortType=1&conditions=searchWord%2B%E5%8C%97%E4%BA%AC%2B1%2B%E5%8C%97%E4%BA%AC"
                        driver.get(url)
                        # 点击登录

                        # 成功篇数
                        crawlSp.append(0)
                        # 失败篇数
                        crawlFp.append(0)
                        # 总篇数
                        crawlTotal.append(crawlnum)

                        while 1:
                            try:
                                driver.find_element_by_class_name("login-btn").click()
                                print("输入密码ing")
                                time.sleep(2)
                                driver.find_element_by_xpath("//input[@id='username']").click()
                                driver.find_element_by_xpath("//input[@id='username']").clear()
                                driver.find_element_by_xpath("//input[@id='username']").send_keys('13667272850')
                                driver.find_element_by_xpath("//input[@id='password']").click()
                                driver.find_element_by_xpath("//input[@id='password']").clear()
                                driver.find_element_by_xpath("//input[@id='password']").send_keys('mssjwow123')
                                driver.find_element_by_class_name("submit").click()
                                time.sleep(2)

                                driver.find_element_by_xpath(
                                    "//input[@placeholder='输入“?”定位到当事人、律师、法官、法院、标题、法院观点']").click()
                                print("搜索关键词ing")
                                driver.find_element_by_xpath(
                                    "//input[@placeholder='输入“?”定位到当事人、律师、法官、法院、标题、法院观点']").clear()
                                driver.find_element_by_xpath(
                                    "//input[@placeholder='输入“?”定位到当事人、律师、法官、法院、标题、法院观点']").send_keys(
                                    keyword)
                                driver.find_element_by_class_name("search-box-btn").click()
                                time.sleep(3)

                                # 加载更多
                                sum = startindex + crawlnum
                                i = int(sum / 20) + 1
                                j = 1
                                for j in range(i):
                                    element = driver.find_element_by_xpath("//button[@class='view-more ng-scope']")
                                    element.click()
                                    time.sleep(3)

                                lis = driver.find_elements_by_xpath(
                                    '//div[@class = "judgements"]/div[@class="judgement ng-scope"]')

                                for i in range(crawlnum):
                                    print("开始点击")
                                    i = i + 1 + startindex
                                    print("在这里")
                                    div_str = '//div[@class="judgements"]/div[{}]/div[2]/h3/a'.format(i)
                                    # title
                                    title = driver.find_element_by_xpath(div_str).text
                                    hl = hashlib.md5()
                                    hl.update(title.encode(encoding='utf-8'))
                                    # md5
                                    title_md5 = hl.hexdigest()

                                    # 将文书存入数据库
                                    instrument_obj = models.Instrument.objects.create(i_title=title, i_path=title_md5, cr_keyword=keyword)
                                    #添加与记录的对应
                                    ci = models.Ciconnect.objects.create(cr_id=now_crid, i_id=instrument_obj.i_id)

                                    # models.Crawl.objects.create(c_keyword=keyword)
                                    # models.Crawl.objects.create(c_title=title)
                                    # models.Crawl.objects.create(c_path=title_md5)
                                    div_str = '//div[@class="judgements"]/div[{}]/div[2]/h3/a'.format(i)
                                    driver.find_element_by_xpath(div_str).click()
                                    print("点击完成")
                                    all_h = driver.window_handles
                                    driver.switch_to.window(all_h[1])
                                    h2 = driver.current_window_handle
                                    print('已定位到元素')
                                    time.sleep(3)
                                    try:
                                        wenshu = driver.find_element_by_xpath(
                                            '//section[@class="paragraphs ng-isolate-scope"]').text
                                        f = open('./data/' + title_md5 + '.txt', 'a')
                                        f.write(wenshu)
                                        f.write('\n')
                                        f.close()
                                        print("成功")
                                        crawlSp[list_index] = crawlSp[list_index] + 1
                                    except:
                                        print("失败")
                                        crawlFp[list_index] = crawlFp[list_index] + 1

                                    driver.close()
                                    driver.switch_to.window(all_h[0])

                                driver.close()

                                json['resultCode'] = '10001'
                                json['resultDesc'] = '操作成功'
                                print(crawlSp[list_index])
                                print(crawlFp[list_index])
                                print(crawlTotal[list_index])
                                print('关闭')
                                # end = time.process_time()
                                break
                            except:
                                print("还未定位到元素!")
                if json['resultCode'] == '10001':
                    now_cr.cr_status = 1
                else:
                    now_cr.cr_status = 2
                now_cr.save()

            except:
                json['resultCode'] = '30000'
                json['resultDesc'] = '服务器故障'

    return JsonResponse(json)

def crawlRate(cr_id):
    index = crawlId.index(cr_id)
    if needcrawl[index]:
        rate = crawlSp[index] / (crawlTotal[index] - crawlFp[index])
    else:
        rate = 1
    return rate

#上传文件
def uploadfile(request):
    print('进入接口uploadfile')
    json = {}

    if request.method == "POST":
        name = request.POST.get("name")
        u_id = 1
        # u_id = request.session.get('u_id')
        name_list = name.split(',')
        data = []

        try:
            for nl in name_list:
                path = './data/' + nl
                # destination = open(path, 'wb+')
                # for chunk in f.chunks():
                #     destination.write(chunk)
                # destination.close()
                time_normal = datetime.datetime.now()
                time_str = str(time_normal).split('.')[0]
                nowtime = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                file_obj = models.File.objects.create(u_id=u_id, f_name=nl, f_path=path, f_time=nowtime)
                rdata = {}
                rdata['f_id'] = file_obj.f_id
                rdata['f_name'] = file_obj.f_name
                data.append(rdata)
            json['resultCode'] = '10001'
            json['resultDesc'] = '操作成功'
            json['data'] = data
        except:
            json['resultCode'] = '30000'
            json['resultDesc'] = '服务器故障'

    return JsonResponse(json)

#获取卡片爬取任务状态
def getcardcr(request):
    print('进入接口getcardcr')
    json = {}

    if request.method == "POST":
        # 返回所有check为0的记录
        u_id = 1
        # u_id = request.session.get('u_id')
        try:
            cr_obj = models.Crawlrecord.objects.filter(u_id=u_id, cr_check=0)
            print(cr_obj.count())
            data = []
            for cr in cr_obj:
                tmp = {}
                cr_id = cr.cr_id
                tmp['cr_id'] = cr_id
                tmp['cr_keyword'] = cr.cr_keyword
                tmp['cr_num'] = cr.cr_num
                tmp['cr_time'] = cr.cr_time
                print('ok')
                if cr.cr_status == 0:
                    rate = crawlRate(cr_id)
                    tmp['cr_flag'] = rate
                else:
                    tmp['cr_flag'] = cr.cr_status
                data.append(tmp)
            json['resultCode'] = '10001'
            json['resultDesc'] = '操作成功'
            json['data'] = data
        except:
            json['resultCode'] = '30000'
            json['resultDesc'] = '服务器故障'

    return JsonResponse(json)

#清除卡片爬取任务记录
def removecardcr(request):
    print('进入接口removecardcr')
    json = {}

    if request.method == "POST":
        u_id = 1
        # u_id = request.session.get('u_id')
        try:
            cr_list_1 = models.Crawlrecord.objects.filter(u_id=u_id, cr_check=0, cr_status=1)
            for cr in cr_list_1:
                cr.cr_check = 1
                cr.save()
            cr_list_2 = models.Crawlrecord.objects.filter(u_id=u_id, cr_check=0, cr_status=2)
            for cr in cr_list_2:
                cr.cr_check = 1
                cr.save()
            cr_obj_0 = models.Crawlrecord.objects.filter(u_id=u_id, cr_check=0)
            data = []
            for cr in cr_obj_0:
                tmp = {}
                cr_id = cr.cr_id
                tmp['cr_id'] = cr_id
                tmp['cr_keyword'] = cr.cr_keyword
                tmp['cr_num'] = cr.cr_num
                tmp['cr_time'] = cr.cr_time
                if cr.cr_status == 0:
                    rate = crawlRate(cr_id)
                    tmp['cr_flag'] = rate
                else:
                    tmp['cr_flag'] = cr.cr_status
                data.append(tmp)
            json['resultCode'] = '10001'
            json['resultDesc'] = '操作成功'
            json['data'] = data
        except:
            json['resultCode'] = '30000'
            json['resultDesc'] = '服务器故障'

    return JsonResponse(json)

#获取卡片分析任务列表
def getcardq(request):
    print('进入接口getcardq')
    json = {}

    if request.method == "POST":
        u_id = 1
        # u_id = request.session.get('u_id')
        print(u_id)
        try:
            # 先获取所有分析任务
            q_objs = models.Question.objects.filter(u_id=u_id, q_check=0)
            crawlidset = set()
            fileidset = set()
            data = []
            for q in q_objs:
                rdata = {}
                tmp_flag = q.r_flag
                # 若为添加相应任务记录id
                if tmp_flag == 0:
                    crawlidset.add(q.r_id)
                else:
                    fileidset.add(q.r_id)
            data = []
            if not not crawlidset:
                for cid in crawlidset:
                    rdata = {}
                    cr = models.Crawlrecord.objects.get(cr_id=cid)
                    rdata['r_id'] = cid
                    rdata['r_flag'] = 0
                    rdata['name'] = cr.cr_keyword
                    rdata['time'] = cr.cr_time
                    question_objs = models.Question.objects.filter(r_flag=0, r_id=cid)
                    questions = [qt.q_name for qt in question_objs]
                    rdata['question'] = questions
                    rdata['q_status'] = question_objs.first().q_status
                    data.append(rdata)
            if not not fileidset:
                for fid in fileidset:
                    fr = models.Filerecord.objects.get(fr_id=fid)
                    f_ids = fr.f_id.split(',')
                    file = models.File.objects.get(f_id=int(f_ids[0]))
                    rdata['r_id'] = fid
                    rdata['r_flag'] = 1
                    rdata['name'] = file.f_name
                    rdata['time'] = fr.fr_time
                    question_objs = models.Question.objects.filter(r_flag=1, r_id=fid)
                    questions = [qt.q_name for qt in question_objs]
                    rdata['question'] = questions
                    rdata['q_status'] = question_objs.first().q_status
                    data.append(rdata)

            json['resultCode'] = '10001'
            json['resultDesc'] = '操作成功'
            json['data'] = data
        except:
            json['resultCode'] = '30000'
            json['resultDesc'] = '服务器故障'

    return JsonResponse(json)

#清除卡片分析任务记录
def removecardq(request):
    print('进入接口removecardq')
    json = {}

    if request.method == "POST":
        u_id = 1
        # u_id = request.session.get('u_id')
        print(u_id)
        try:
            # 获取已完成分析任务
            q_1_objs = models.Question.objects.filter(u_id=u_id, q_check=0, q_status=1)
            for q_1 in q_1_objs:
                q_1.q_check = 1
                q_1.save()
            # 获取失败分析任务
            q_2_objs = models.Question.objects.filter(u_id=u_id, q_check=0, q_status=2)
            for q_2 in q_2_objs:
                q_2.q_check = 1
                q_2.save()
            q_objs = models.Question.objects.filter(u_id=u_id, q_check=0)
            crawlidset = set()
            fileidset = set()
            data = []
            for q in q_objs:
                rdata = {}
                tmp_flag = q.r_flag
                # 若为添加相应任务记录id
                if tmp_flag == 0:
                    crawlidset.add(q.r_id)
                else:
                    fileidset.add(q.r_id)
            data = []
            if not not crawlidset:
                for cid in crawlidset:
                    rdata = {}
                    cr = models.Crawlrecord.objects.get(cr_id=cid)
                    rdata['r_id'] = cid
                    rdata['r_flag'] = 0
                    rdata['name'] = cr.cr_keyword
                    rdata['time'] = cr.cr_time
                    question_objs = models.Question.objects.filter(r_flag=0, r_id=cid)
                    questions = [qt.q_name for qt in question_objs]
                    rdata['question'] = questions
                    rdata['q_status'] = question_objs.first().q_status
                    data.append(rdata)
            if not not fileidset:
                for fid in fileidset:
                    fr = models.Filerecord.objects.get(fr_id=fid)
                    f_ids = fr.f_id.split(',')
                    file = models.File.objects.get(f_id=int(f_ids[0]))
                    rdata['r_id'] = fid
                    rdata['r_flag'] = 1
                    rdata['name'] = file.f_name
                    rdata['time'] = fr.fr_time
                    question_objs = models.Question.objects.filter(r_flag=1, r_id=fid)
                    questions = [qt.q_name for qt in question_objs]
                    rdata['question'] = questions
                    rdata['q_status'] = question_objs.first().q_status
                    data.append(rdata)

            json['resultCode'] = '10001'
            json['resultDesc'] = '操作成功'
            json['data'] = data
        except:
            json['resultCode'] = '30000'
            json['resultDesc'] = '服务器故障'

    return JsonResponse(json)

#任务中心获取用户爬虫任务记录
def getcrawlrecord(request):
    print('进入接口getcrawlrecord')
    json = {}

    if request.method == "POST":
        u_id = 1
        # u_id = request.session.get('u_id')
        try:
            cr_obj = models.Crawlrecord.objects.filter(u_id=u_id)
            data = []
            for cr in cr_obj:
                tmp = {}
                cr_id = cr.cr_id
                tmp['cr_id'] = cr_id
                tmp['cr_keyword'] = cr.cr_keyword
                tmp['cr_num'] = cr.cr_num
                tmp['cr_time'] = cr.cr_time
                if cr.cr_status == 0:
                    rate = crawlRate(cr_id)
                    tmp['cr_flag'] = rate
                else:
                    tmp['cr_flag'] = cr.cr_status
                data.append(tmp)
            json['resultCode'] = '10001'
            json['resultDesc'] = '操作成功'
            json['data'] = data
        except:
            json['resultCode'] = '30000'
            json['resultDesc'] = '服务器故障'

    return JsonResponse(json)

#任务中心获取用户文件上传记录
def getfiles(request):
    print('进入接口getfiles')
    json = {}

    if request.method == "POST":
        u_id = 1
        # u_id = request.session.get('u_id')
        try:
            files = models.File.objects.filter(u_id=u_id).values()
            json['resultCode'] = '10001'
            json['resultDesc'] = '操作成功'
            json['data'] = list(files)
        except:
            json['resultCode'] = '30000'
            json['resultDesc'] = '服务器故障'

    return JsonResponse(json)

#分析页面获取用户已成功爬虫记录
def analyzeCR(request):
    print('进入接口analyzeCR')
    json = {}

    if request.method == "POST":
        u_id = 1
        # u_id = request.session.get('u_id')
        try:
            print('进入')
            analyze_cr_list = models.Crawlrecord.objects.filter(u_id=u_id, cr_status=1).values()
            json['resultCode'] = '10001'
            json['resultDesc'] = '操作成功'
            json['data'] = list(analyze_cr_list)
            print('成功')
        except:
            print('失败')
            json['resultCode'] = '30000'
            json['resultDesc'] = '服务器故障'

    return JsonResponse(json)

#获取任务中心分析任务记录
def getanalyzerecord(request):
    print('进入接口getanalyzerecord')
    json = {}

    if request.method == "POST":
        u_id = 1
        # u_id = request.session.get('u_id')
        print(u_id)
        try:
            # 先获取所有分析任务
            q_objs = models.Question.objects.filter(u_id=u_id)
            crawlidset = set()
            fileidset = set()
            data = []
            for q in q_objs:
                tmp_flag = q.r_flag
                # 若为添加相应任务记录id
                if tmp_flag == 0:
                    crawlidset.add(q.r_id)
                else:
                    fileidset.add(q.r_id)
            if not not crawlidset:
                for cid in crawlidset:
                    rdata = {}
                    cr = models.Crawlrecord.objects.get(cr_id=cid)
                    rdata['r_id'] = cid
                    rdata['r_flag'] = 0
                    rdata['name'] = cr.cr_keyword
                    rdata['time'] = cr.cr_time
                    question_objs = models.Question.objects.filter(r_flag=0, r_id=cid)
                    questions = [qt.q_name for qt in question_objs]
                    first = models.Question.objects.filter(r_flag=0, r_id=cid).first()
                    rdata['q_status'] = first.q_status
                    rdata['question'] = questions
                    data.append(rdata)
            if not not fileidset:
                for fid in fileidset:
                    rdata = {}
                    fr = models.Filerecord.objects.get(fr_id=fid)
                    f_ids = fr.f_id.split(',')
                    print(f_ids)
                    print(f_ids[0])
                    file = models.File.objects.get(f_id=int(f_ids[0]))
                    rdata['r_id'] = fid
                    rdata['r_flag'] = 1
                    rdata['name'] = file.f_name
                    rdata['time'] = fr.fr_time
                    question_objs = models.Question.objects.filter(r_flag=1, r_id=fid)
                    questions = [qt.q_name for qt in question_objs]
                    rdata['question'] = questions
                    print('ok2')
                    first = models.Question.objects.filter(r_flag=1, r_id=fid).first()
                    rdata['q_status'] = first.q_status
                    data.append(rdata)

            json['resultCode'] = '10001'
            json['resultDesc'] = '操作成功'
            json['data'] = data
        except:
            json['resultCode'] = '30000'
            json['resultDesc'] = '服务器故障'

    return JsonResponse(json)

# def readcomprehend(request):
#     print('进入接口readcomprehend')
#     json = {}
#
#     try:
#         if request.method == "POST":
#             # questions = request.POST.get("questions")
#             r_flag = int(request.POST.get("r_flag"))
#             r_id = request.POST.get("r_id")
#
#             u_id = 1
#             # u_id = request.session.get('u_id')
#             # question_list通过questions用逗号划分
#             instrument_id_list = []
#             data = []
#
#             # 如果是爬虫分析
#             if r_flag == 0:
#                 data.append(r_id)
#                 models.Ciconnect.objects.filter(cr_id=int(r_id)).delete()
#                 i_title_list = ['朱某某滥伐林木案一审刑事判决书', '王某富滥伐林木一审刑事判决书', '龙进前、龙天豪滥伐林木一审刑事判决书',
#                                 '陆大锴、龙先堂滥伐林木案一审刑事判决书', '黄某滥伐林木一审刑事判决书', '范烈辉滥伐林木案一审刑事判决书',
#                                 '罗某辉滥伐林木一审刑事判决书', '刘光来滥伐林木案一审刑事判决书', '锦屏县启蒙镇归固村某村民小组、罗某进滥伐林木一审刑事判决书',
#                                 '叶某爽滥伐林木罪一案一审刑事判决书']
#                 q_list = ['刑罚种类是什么', '涉案金额多少', '刑期多久']
#                 answer_list = []
#                 a1_list = ['有期徒刑、罚金', '有期徒刑、罚金', '期徒刑、罚金', '拘役、罚金', '有期徒刑、罚金', '拘役、罚金', '有期徒刑、罚金', '有期徒刑、罚金', '有期徒刑、罚金', '有期徒刑、罚金']
#                 a2_list = ['三万二千元', '七千六百四十一元', '八千一百九十九元', '一万五千元', '5000元', '六千一百零八元', '一万二千二百五十元', '九千元', '三千元', '三万元']
#                 a3_list = ['三年', '一年零六个月', '一年', '五个月', '二年', '三个月', '一年', '一年六个月', '一年', '三年']
#                 answer_list.append(a1_list)
#                 answer_list.append(a2_list)
#                 answer_list.append(a3_list)
#                 for i in range(10):
#                     path = './data' + i_title_list[i] + '.docx'
#                     print('ok')
#                     instrument = models.Instrument.objects.create(i_title=i_title_list[i], i_path=path,
#                                                                   cr_keyword='环境资源犯罪')
#                     instrument_id_list.append(instrument.i_id)
#                     models.Ciconnect.objects.create(cr_id=int(r_id), i_id=instrument.i_id)
#                 for i in range(3):
#                     q_obj = models.Question.objects.create(q_name=q_list[i], r_id=int(r_id), r_flag=r_flag, u_id=u_id, q_status=0,
#                                                    q_check=0)
#                     for j in range(10):
#                         models.Answer.objects.create(a_answer=answer_list[i][j], q_id=q_obj.q_id,i_id=instrument_id_list[j])
#
#             # 如果是上传文件分析，id为需要分析文件的集合
#             else:
#                 # 先创建分析记录
#                 time_normal = datetime.datetime.now()
#                 time_str = str(time_normal).split('.')[0]
#                 fr_time = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
#                 answer_list = []
#                 q_list = ['案例涉及法条有哪些', '涉案金额多少', '判决罪名有哪些']
#                 a1_list = ['《中华人民共和国刑法》第二百六十六条、第三百一十二条、第二十五条第一款、第二十六条、第二十七条、第六十七条、'
#                                '第六十八条；《中华人民共和国刑法》第二百六十六条、第三百一十二条第一款、第二十五条第一款、第二十六条第一、'
#                                '四款、第二十七条、第六十七条第一款、第六十八条、第七十二条第一、三款、第六十四条',
#                                '《中华人民共和国行政诉讼法》（以下简称行政诉讼法）第四十九条第（三）项、《关于适用<中华人民共和国行政'
#                                '诉讼法>若干问题的解释》（以下简称适用问题解释）第三条第一款第（一）项、行政诉讼法第四十九条第（三）项、'
#                                '《电信用户申诉处理暂行办法》第四条及第六条、《中华人民共和国行政诉讼法》第八十九条第一款第（一）项',
#                            '《中华人民共和国民事诉讼法》第二百零四条第一款、《最高人民法院关于适用<中华人民共和国民事诉讼法>的解释》第三百九十五条第二款']
#                 a2_list = ['一百三十六万五千元', '', '']
#                 a3_list = ['诈骗罪、掩饰、隐瞒犯罪所得罪', '', '']
#                 answer_list.append(a1_list)
#                 answer_list.append(a2_list)
#                 answer_list.append(a3_list)
#                 file_name = ['（2019）浙0602刑初216号刑事判决书', '白燕与工业和信息化部其他二审行政裁定书', '李航借记卡纠纷申诉、申请民事裁定书']
#                 file_id_list = r_id.split(',')
#                 for fid in file_id_list:
#                     models.File.objects.get(f_id=fid).delete()
#                 file_id_list = []
#                 for i in range(3):
#                     file = models.File.objects.create(u_id=u_id, f_name=file_name[i], f_path='./data', f_time=fr_time)
#                     file_id_list.append(file.f_id)
#                 idstr = ",".join(str(i) for i in file_id_list)
#                 fr = models.Filerecord.objects.create(f_id=idstr, fr_time=fr_time)
#                 data.append(fr.fr_id)
#                 for i in range(3):
#                     q_obj = models.Question.objects.create(q_name=q_list[i], r_id=fr.fr_id, r_flag=r_flag, u_id=u_id, q_status=0,
#                                                    q_check=0)
#                     for j in range(3):
#                         models.Answer.objects.create(a_answer=answer_list[i][j], q_id=q_obj.q_id, i_id=file_id_list[j])
#             json['resultCode'] = '10001'
#             json['resultDesc'] = '操作成功'
#             json['data'] = data
#     except:
#         json['resultCode'] = '30000'
#         json['resultDesc'] = '服务器故障'
#
#     return JsonResponse(json)

def readcomprehend(request):
    print('进入接口readcomprehend')
    json = {}

    if request.method == "POST":
        questions = request.POST.get("questions")
        keyword = request.POST.get("keyword")
        #print(questions)
        try:
            k_id = models.Keyword.objects.get(k_keyword=keyword)
        except:
            json['resultCode'] = '30000'
            json['resultDesc'] = '服务器故障'

        # question_list通过questions用分号划分
        question_list = questions.split(";")
        question_id_list = []

        for question in question_list:
            try:
                questionExist = models.Question.objects.get(q_name=question, k_id=k_id)
                q_id = questionExist.q_id
            except:
                q_id = models.Question.objects.create(q_name=question, k_id=k_id).q_id
            question_id_list.append(q_id)

        #查询question_id
        # question_id_list2 = models.Question.objects.filter(k_id=k_id).values_list('q_id')
        #print ("question_id_list2:", question_id_list2)
        question_id_list = [one_id[0] for one_id in question_id_list]

        # 1，根据当前的关键字id查询出question列表，question的id列表，篇章列表，篇章的id列表
       # passage_list, passage_id_list = models.Crawl.objects.filter(k_id=k_id).values_list('c_id', 'c_path')
        all_passage_list= models.Crawl.objects.filter(k_id=k_id).values_list('c_id', 'c_path')
        passage_list2 = []
        passage_id_list = []
        for one_passage_list in all_passage_list:
            passage_id_list.append(one_passage_list[0])
            passage_list2.append(one_passage_list[1])
        #print ("查询结束")

        passage_list = []
        for i in range(len(passage_list2)):
            file = open("./data/"+passage_list2[i]+".txt", "r")#, encoding="gbk"
            one_passage = file.read()
            file.close()
            passage_list.append(one_passage)
        #print (passage_list2[0])
        #print ("成功")

        # 2，得到四个列表之后开始分析
        # 下面为四个list的例子
        """
        file = open("legalReadFunc/data/wenshu.txt", "r", encoding="utf8")
        data = file.read()
        passage_list = data.split("\n\n\n\n")
        passage_id_list = range(len(passage_list))
        question_id_list = range(len(question_list))
        question_list = ["罪名是什么？", "刑期有多久？", "涉案金额是多少？", "作案人数有几人？"]
        """

        # 3，进行分析
        all_predictions = main.main2(passage_list, question_list)

        # 4，整理分析结果
        return_data = []
        for q_id in all_predictions.keys():
            one_return = {}
            position = q_id.split("_")
            one_return["passage_id"] = passage_id_list[int(position[0])]
            one_return['question_id'] = question_id_list[int(position[1])]
            one_return['answer'] = all_predictions[q_id]
            return_data.append(one_return)

        print(return_data)

    #把return_data存到answer表中
    data_list = []
    flag = True
    for a in return_data:
        models.Answer.objects.create(a_answer=a['answer'], q_id=a['question_id'], c_id=a['passage_id'], k_id=k_id)
        try:
            c_title = models.Crawl.objects.get(c_id=a['passage_id']).c_title
            q_name = models.Question.objects.get(q_id=a['question_id']).q_name
            data_dict = {}
            data_dict['question'] = q_name
            data_dict['answer'] = a['answer']
            data_dict['passage'] = c_title
            data_list.append(data_dict)
        except:
            json['resultCode'] = '30000'
            json['resultDesc'] = '服务器故障'
            flag = False

    if flag:
        json['resultCode'] = '10001'
        json['resultDesc'] = '操作成功'
        json['data'] = data_list

    return JsonResponse(json)

def change(request):
    print("进入接口readcomprehend")
    json = {}

    if request.method == "POST":
        r_flag = int(request.POST.get("r_flag"))
        r_id = int(request.POST.get("r_id"))

        time.sleep(60)

        models.Question.objects.filter(r_id=r_id, r_flag=r_flag, q_status=0).update(q_status=1)

        json['resultCode'] = '10001'
        json['resultDesc'] = '操作成功'

    return JsonResponse(json)

def getanswer(request):
    print("进入接口getanswer")
    json = {}

    if request.method == "POST":
        #记录是爬虫数据分析，还是上传数据分析
        r_flag = int(request.POST.get("r_flag"))
        r_id = int(request.POST.get("r_id"))
        try:
            data = []
            #若是查询爬虫分析任务
            if r_flag == 0:
                ci = models.Ciconnect.objects.filter(cr_id=r_id)
                question = models.Question.objects.filter(r_id=r_id, r_flag=0)
                for one_ci in ci:
                    instrument = models.Instrument.objects.get(i_id=one_ci.i_id)
                    for q in question:
                        rdata = {}
                        answer = models.Answer.objects.get(q_id=q.q_id, i_id=instrument.i_id)
                        rdata['question'] = q.q_name
                        rdata['filename'] = instrument.i_title
                        rdata['answer'] = answer.a_answer
                        data.append(rdata)
            #若是上传文件分析任务
            else:
                fr = models.Filerecord.objects.get(fr_id=r_id)
                fileid = fr.f_id.split(',')
                print(fileid)
                question = models.Question.objects.filter(r_id=r_id, r_flag=1)
                for flid in fileid:
                    file = models.File.objects.get(f_id=flid)
                    for q in question:
                        rdata = {}
                        answer = models.Answer.objects.get(q_id=q.q_id, i_id=flid)
                        print(answer)
                        rdata['question'] = q.q_name
                        rdata['filename'] = file.f_name
                        rdata['answer'] = answer.a_answer
                        data.append(rdata)

            json['resultCode'] = '10001'
            json['resultDesc'] = '操作成功'
            json['data'] = data

        except:
            json['resultCode'] = '30000'
            json['resultDesc'] = '服务器故障'

    return JsonResponse(json)