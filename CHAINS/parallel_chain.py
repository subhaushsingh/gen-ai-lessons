from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableParallel

load_dotenv()



llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    
)

model1 = ChatHuggingFace(llm=llm)

model2 = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

prompt1 = PromptTemplate(
    template="generate notes short and simple for the text -> \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="generate 5 questions from the text -> \n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="merge the text and the quiz into single document-> \n {text} and {quiz}",
    input_variables=['text','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'text' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})


merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = '''## A Report on Cricket: An Enduring Sport in the Modern Era

**1. Introduction:**
Cricket, a bat-and-ball game originating in England, stands as one of the world's most popular sports by global popularity and attracts billions of followers each year. From passionate fans in test matches to informal matches in the street, cricket boasts an undeniable charm and complex structure both on and off the pitch.

**2. Playing Structure:**
Crudely compared to a hybrid of baseball and football, cricket has evolved over centuries with varying playing formats. Here are the most prevalent:

   * **Test Cricket:** The longest format, each match lasting up to 5 days, focuses on prolonged battles between batting teams and disciplined bowling attacks. Test matches are often considered the pinnacle of play, showcasing strategic depth and relentless test of skill.
   * **One Day International (ODI):** A condensed format played over a single day, ODI focuses on strategic runscoring during limited overs. Team's utilize variation in pace, spin, and fielding to secure a substantial lead or win, emphasizing rapid, high-quality batting and spirited bowling.
   * **Twenty20 (T20):** The entertainment-focused flagship format, T20 displays explosive innings and rapid reflexes. Players showcase their aggressive talent and adapt easily to short bursts of potential.

**3. Core Elements of the Game:**
Each format shares some key elements, but the specific skill sets they demand define the unique demands of each.
   * **Batting:**  Players aim to score runs by hitting the ball and running between wickets strategically. Focus lies on technical prowess, timing, and adapting to different bowling types.
   * **Bowling:**  The role of bowlers, a crucial component of any format, involves using various deliveries to evade batsmen, deceive their technique, and achieve valuable wickets.  Delivery variations and accuracy are paramount.

**4. Historical Significance:**
Originating in England during the 16th century, cricket experienced early growth in colonies and popularized throughout the 19th century. The English team's victories in the Ashes series in the late 19th and early 20th century cemented cricketing dominance, leading to the formation of international sporting regulations and organizations that are essential for the sport's continuity.

**5. Global Influence:**
Today, cricket is not only a beloved sport but a cultural phenomenon. The sport connects nations, fuels economies, and unites fans across the globe.

   * **Impact on Economies:**  Cricket stadiums and tournament venues generate significant income through attendances and broadcast rights. Fan engagement through merchandise, sponsorship, and broadcasting drives economic growth, particularly in emerging markets like India and Africa.
   * **Connecting Cultures:**  The sport fosters cultural exchange and awareness abroad through the travel and participation of international teams and players, leading to deepened understanding and appreciation of diverse cultures.

**6. Challenges and Future Prospects:**
While cricket enjoys an undeniable global appeal, the sport faces fresh and complex challenges, including:


   * **Reaching Younger Generations:**  Finding excitement amongst younger audiences all around the world is critical as it ensures the sport's long-term success, engaging multi-generational audiences with innovative strategies to generate excitement.
   * **Gender Parity and Representation:**  Expanding female participation across all levels of the sport remains crucial for achieving equality and equity and modernizing the sport's structure for diverse communities.

**7. Conclusion:**
Cricket, despite its traditional roots, is a dynamic and evolving sport with an enduring embrace across the global community. From its strategic depths and complex rules to its captivating victories, cricket continues to resonate with fans as players and officials strive to optimize its growth and nurture generational interest in the sport for years to come.



**8.  Appendix:**

   * **World Cricket Governing Council (ICC) and its key functions:** The ICC establishes and governs international rules, conducts various tournaments, and especially promotes investment in its member nations.
   * **Famous Cricketers:** Past and present figures shape cricket's legacy through both skill and cultural influence. A list of notable figures could be included, detailing their significant contributions.
   * **Major Cricket Events:** From the prestigious County Championship to the spectacle of the Cricket World Cup,  highlighting major holistic events showcasing diverse regions, nations, and teams.


By using this template, you can customize and expand with your own insights about specific leagues, players, historical wins, or even emerging trends and technologies in cricket.'''

result = chain.invoke({'text':text})


print(result)