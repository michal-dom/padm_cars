from urllib.request import Request, urlopen
import re

otomoto_link = "https://www.otomoto.pl/osobowe/audi/a4/b8-2007-2015/seg-combi/?search%5Bbrand_program_id%5D%5B0%5D=&search%5Bcountry%5D=&page="
file_name = "audi_links.txt"

def link_getter(page_url):
    # fp = urllib.request.urlopen(page_url)
    # mybytes = fp.read()
    # print(page_url)

    req = Request(
        page_url,
        headers={'User-Agent': 'Mozilla/5.0'})

    mybytes = urlopen(req).read()

    mystr = mybytes.decode("utf8")
    reg = re.findall(r"<a[^>]* href=\"([^\"]*)\"",
        mystr,
        re.DOTALL)
    for link in reg:
        if "oferta" in link:
            with open(file_name, "a") as myfile:
                myfile.write(link)
                myfile.write("\n")
            print(link)

for i in range(1, 52):

    link_getter(
        otomoto_link + str(i))

# link_getter("https://www.otomoto.pl/osobowe/volkswagen/golf/v-2003-2009/?search%5Bbrand_program_id%5D%5B0%5D=&search%5Bcountry%5D=")