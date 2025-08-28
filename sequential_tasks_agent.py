from axiestudio.components.crewai.sequential_crew import SequentialCrewComponent
from axiestudio.components.crewai.sequential_task_agent import SequentialTaskAgentComponent
from axiestudio.components.input_output import ChatOutput, TextInputComponent
from axiestudio.components.openai.openai_chat_model import OpenAIModelComponent
from axiestudio.components.processing import PromptComponent
from axiestudio.components.tools import SearchAPIComponent
from axiestudio.graph import Graph


def sequential_tasks_agent_graph():
    llm = OpenAIModelComponent()
    search_api_tool = SearchAPIComponent()

    text_input = TextInputComponent(_display_name="Topic")
    text_input.set(input_value="Agile")

    # Dokumentprompt för forskare
    document_prompt_component = PromptComponent()
    document_prompt_component.set(
        template="""Ämne: {topic}

Bygg ett dokument om detta ämne.""",
        topic=text_input.text_response,
    )

    # Forskaruppgiftsagent
    researcher_task_agent = SequentialTaskAgentComponent()
    researcher_task_agent.set(
        role="Forskare",
        goal="Sök på Google för att hitta information för att slutföra uppgiften.",
        backstory="Forskning har alltid varit din grej. Du kan snabbt hitta saker på webben på grund av dina färdigheter.",
        tools=[search_api_tool.build_tool],
        llm=llm.build_model,
        task_description=document_prompt_component.build_prompt,
        expected_output="Punktlistor och små fraser om forskningsämnet.",
    )

    # Revisionsprompt för redaktör
    revision_prompt_component = PromptComponent()
    revision_prompt_component.set(
        template="""Ämne: {topic}

Revidera detta dokument.""",
        topic=text_input.text_response,
    )

    # Redaktörsuppgiftsagent
    editor_task_agent = SequentialTaskAgentComponent()
    editor_task_agent.set(
        role="Redaktör",
        goal="Du bör redigera informationen som tillhandahålls av forskaren för att göra den mer smaklig och för att inte innehålla "
        "vilseledande information.",
        backstory="Du är redaktör för den mest ansedda tidskriften i världen.",
        llm=llm.build_model,
        task_description=revision_prompt_component.build_prompt,
        expected_output="Små stycken och punktlistor med det korrigerade innehållet.",
        previous_task=researcher_task_agent.build_agent_and_task,
    )

    # Bloggprompt för komiker
    blog_prompt_component = PromptComponent()
    blog_prompt_component.set(
        template="""Ämne: {topic}

Bygg ett roligt blogginlägg om detta ämne.""",
        topic=text_input.text_response,
    )

    # Komikeruppgiftsagent
    comedian_task_agent = SequentialTaskAgentComponent()
    comedian_task_agent.set(
        role="Komiker",
        goal="Du skriver komiskt innehåll baserat på informationen som tillhandahålls av redaktören.",
        backstory="Din formella yrkestittel är Chefskomiker. "
        "Du skriver skämt, gör ståuppkomedi och skriver roliga artiklar.",
        llm=llm.build_model,
        task_description=blog_prompt_component.build_prompt,
        expected_output="En liten blogg om ämnet.",
        previous_task=editor_task_agent.build_agent_and_task,
    )

    crew_component = SequentialCrewComponent()
    crew_component.set(
        tasks=comedian_task_agent.build_agent_and_task,
    )

    # Ställ in utdatakomponenten
    chat_output = ChatOutput()
    chat_output.set(input_value=crew_component.build_output)

    # Skapa grafen
    return Graph(
        start=text_input,
        end=chat_output,
        flow_name="Sekventiell Uppgiftsagent",
        description="Denna agent kör uppgifter i en fördefinierad sekvens.",
    )
