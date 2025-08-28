from axiestudio.components.crewai.crewai import CrewAIAgentComponent
from axiestudio.components.crewai.hierarchical_crew import HierarchicalCrewComponent
from axiestudio.components.crewai.hierarchical_task import HierarchicalTaskComponent
from axiestudio.components.input_output import ChatInput, ChatOutput
from axiestudio.components.openai.openai_chat_model import OpenAIModelComponent
from axiestudio.components.processing import PromptComponent
from axiestudio.components.tools import SearchAPIComponent, YfinanceToolComponent
from axiestudio.graph import Graph


def complex_agent_graph():
    llm = OpenAIModelComponent(model_name="gpt-4o-mini")
    manager_llm = OpenAIModelComponent(model_name="gpt-4o")
    search_api_tool = SearchAPIComponent()
    yahoo_search_tool = YfinanceToolComponent()
    dynamic_agent = CrewAIAgentComponent()
    chat_input = ChatInput()
    role_prompt = PromptComponent(_display_name="Roll-prompt")
    role_prompt.set(
        template="""Definiera en roll som kan utföra eller svara väl på användarens fråga.

Användarens fråga: {query}

Rollen bör vara max två ord. Något som "Forskare" eller "Mjukvaruutvecklare".
"""
    )

    goal_prompt = PromptComponent(_display_name="Mål-prompt")
    goal_prompt.set(
        template="""Definiera målet för denna roll, givet användarens fråga.
Användarens fråga: {query}

Roll: {role}

Målet bör vara kortfattat och specifikt.
Mål:
""",
        query=chat_input.message_response,
        role=role_prompt.build_prompt,
    )
    backstory_prompt = PromptComponent(_display_name="Bakgrundshistoria-prompt")
    backstory_prompt.set(
        template="""Definiera en bakgrundshistoria för denna roll och mål, givet användarens fråga.
Användarens fråga: {query}

Roll: {role}
Mål: {goal}

Bakgrundshistorian bör vara specifik och väl anpassad till resten av informationen.
Bakgrundshistoria:""",
        query=chat_input.message_response,
        role=role_prompt.build_prompt,
        goal=goal_prompt.build_prompt,
    )
    dynamic_agent.set(
        tools=[search_api_tool.build_tool, yahoo_search_tool.build_tool],
        llm=llm.build_model,
        role=role_prompt.build_prompt,
        goal=goal_prompt.build_prompt,
        backstory=backstory_prompt.build_prompt,
    )

    response_prompt = PromptComponent()
    response_prompt.set(
        template="""Användarens fråga:
{query}

Svara användaren med så mycket information som du kan om ämnet. Ta bort om det behövs.
Om det bara är en allmän fråga (t.ex. en hälsning) kan du svara dem direkt.""",
        query=chat_input.message_response,
    )
    manager_agent = CrewAIAgentComponent()
    manager_agent.set(
        llm=manager_llm.build_model,
        role="Chef",
        goal="Du kan svara på allmänna frågor från användaren och kan kalla på andra för hjälp om det behövs.",
        backstory="Du är artig och hjälpsam. Du har alltid varit en ledstjärna för artighet.",
    )
    task = HierarchicalTaskComponent()
    task.set(
        task_description=response_prompt.build_prompt,
        expected_output="Kortfattat svar som besvarar användarens fråga.",
    )
    crew_component = HierarchicalCrewComponent()
    crew_component.set(
        tasks=task.build_task, agents=[dynamic_agent.build_output], manager_agent=manager_agent.build_output
    )
    chat_output = ChatOutput()
    chat_output.set(input_value=crew_component.build_output)

    return Graph(
        start=chat_input,
        end=chat_output,
        flow_name="Sekventiell uppgiftsagent",
        description="Denna agent kör uppgifter i en fördefinierad sekvens.",
    )
