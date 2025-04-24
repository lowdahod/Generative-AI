from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline("summarization")

# Expanded input passage
text = """
Artificial Intelligence (AI) is transforming education by introducing adaptive learning techniques, automating administrative processes, 
and enabling intelligent tutoring systems. AI-driven learning platforms analyze vast amounts of student data, including learning habits, strengths, 
and weaknesses, to personalize educational experiences. This customization allows students to progress at their own pace, ensuring that they receive content 
suited to their proficiency level. Additionally, AI chatbots and virtual assistants are becoming common in academic institutions, providing real-time support to students. 
These tools answer frequently asked questions, guide students through complex topics, and help with scheduling and reminders. Educators also benefit from AI-powered 
grading systems that assess assignments, quizzes, and exams, significantly reducing workload and providing instant feedback. Moreover, AI enhances accessibility in 
education by offering language translation services, speech-to-text conversion, and assistive technologies for students with disabilities. By breaking language barriers 
and supporting diverse learning needs, AI makes education more inclusive. However, challenges remain in implementing AI in education. Data privacy concerns arise as student 
information is collected and analyzed, requiring robust security measures. There is also the risk of AI biases, where algorithmic decisions may favor certain groups over 
others due to biased training data. Additionally, educators must undergo proper training to integrate AI effectively into their teaching methods. To fully harness AI’s
potential in education, institutions must adopt ethical AI frameworks, ensure transparency in algorithmic decision-making, and continuously update their technological 
infrastructure. Collaboration between educators, policymakers, and AI developers is crucial in shaping the future of education and ensuring that AI serves as an enabler 
rather than a disruptor.
"""

# Generate the summary with longer output
summary = summarizer(text, max_length=100, min_length=50, do_sample=False)

# Print the summarized text
print("Summarized Text:\n", summary[0]['summary_text'])
