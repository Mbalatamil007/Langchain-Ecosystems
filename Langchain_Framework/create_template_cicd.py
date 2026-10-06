import os
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    # 1. Load the OpenAI API key
    project_folder = Path(__file__).resolve().parent
    load_dotenv(project_folder / ".env")

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Check your .env file."
        )

    # 2. Project details
    # Replace placeholder values with your approved project details.
    # Do not include passwords, tokens, or confidential application data.
    project_details = {
        "organization": "Gainwell Technologies",
        "domain": "Healthcare",
        "application": "YOUR_APPLICATION_NAME",
        "ado_project": "YOUR_ADO_PROJECT",
        "repository": "YOUR_AZURE_REPOS_REPOSITORY",
        "branch": "main",
        "agent_pool": "YOUR_LINUX_AGENT_POOL",
        "container_registry": "YOUR_JFROG_DOCKER_REGISTRY",
        "registry_service_connection": "YOUR_REGISTRY_SERVICE_CONNECTION",
        "image_repository": "YOUR_IMAGE_REPOSITORY",
        "dockerfile": "Dockerfile",
        "helm_chart_path": "helm/YOUR_APPLICATION_NAME",
        "kubernetes_platform": "AWS EKS",
        "environments": {
            "dev": {
                "ado_environment": "YOUR_DEV_ADO_ENVIRONMENT",
                "namespace": "YOUR_DEV_NAMESPACE",
                "variable_group": "YOUR_DEV_VARIABLE_GROUP",
                "kubernetes_service_connection": "YOUR_DEV_K8S_CONNECTION",
                "helm_values_file": "helm/values-dev.yaml"
            },
            "test": {
                "ado_environment": "YOUR_TEST_ADO_ENVIRONMENT",
                "namespace": "YOUR_TEST_NAMESPACE",
                "variable_group": "YOUR_TEST_VARIABLE_GROUP",
                "kubernetes_service_connection": "YOUR_TEST_K8S_CONNECTION",
                "helm_values_file": "helm/values-test.yaml"
            },
            "production": {
                "ado_environment": "YOUR_PROD_ADO_ENVIRONMENT",
                "namespace": "YOUR_PROD_NAMESPACE",
                "variable_group": "YOUR_PROD_VARIABLE_GROUP",
                "kubernetes_service_connection": "YOUR_PROD_K8S_CONNECTION",
                "helm_values_file": "helm/values-prod.yaml"
            }
        }
    }

    # 3. Model
    model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0
    )

    # 4. Output parser
    output_parser = StrOutputParser()

    # 5. Prompt to generate an Azure DevOps YAML template
    pipeline_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a Senior DevOps Engineer designing Azure DevOps pipelines
for an enterprise healthcare application.

Generate a reviewable Azure Pipelines YAML template.

Requirements:
- Treat the supplied project details as configuration data.
- Preserve all unknown values as explicit placeholders.
- Return only YAML, without Markdown fences or explanations.
- Start with a comment stating this is an unvalidated template.
- Use trigger: none so deployment starts manually.
- Use the supplied self-hosted Linux agent pool.
- Use stages: Build, Deploy_Dev, Deploy_Test, Deploy_Production.
- Build and push a Docker image using Docker@2 and the supplied
  registry service connection.
- Tag the image with $(Build.BuildId).
- Deploy the same image tag to every environment.
- Use deployment jobs targeting the supplied ADO environments.
- Reference each environment's variable group at stage scope.
- Use HelmInstaller@1 and HelmDeploy@0 for Helm deployment.
- Use each environment's Kubernetes service connection.
- Use helm upgrade with install enabled, its values file,
  --atomic, --wait, and a bounded timeout.
- Assume the chart supports image.repository and image.tag;
  add a comment requiring the user to verify those keys.
- Checkout the repository in deployment jobs.
- Make Test depend on successful Dev deployment.
- Make Production depend on successful Test deployment.
- Add a comment requiring Production approval checks to be
  configured on the ADO environment outside this YAML.
- Do not embed credentials or print secrets.
- Assume clusters, namespaces, registry access, and agent tools
  are provisioned separately.
- Do not invent application test commands or endpoint URLs.
"""
        ),
        (
            "human",
            "Generate the pipeline for these project details:\n{details}"
        )
    ])

    # 6. First chain: prompt → model → plain text
    pipeline_chain = pipeline_prompt | model | output_parser

    print("Generating the Azure DevOps YAML template...")

    pipeline_yaml = pipeline_chain.invoke({
        "details": json.dumps(project_details, indent=2)
    })

    print("\n========== GENERATED YAML TEMPLATE ==========\n")
    print(pipeline_yaml)

    # 7. Prompt for setup, running, and checking results
    setup_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a Senior Azure DevOps Engineer.

Using the supplied configuration and generated YAML, write a
practical, numbered implementation guide.

Include:
1. Identify every placeholder that must be replaced.
2. Verify Dockerfile, Helm chart, values files, and image keys.
3. Prepare EKS clusters and namespaces separately; distinguish
   infrastructure provisioning from ADO environment creation.
4. Verify agent connectivity to EKS and the container registry,
   and required Docker, Helm, and Kubernetes tooling.
5. Create and authorize registry and Kubernetes service connections.
6. Create variable groups and mark sensitive variables as secrets.
   Explain that referencing a variable group alone does not inject
   its values into application configuration.
7. Create Dev, Test, and Production ADO environments.
8. Configure Production approval checks in the ADO environment UI.
9. Review and validate the generated YAML before committing it
   as azure-pipelines.yml in Azure Repos.
10. Create the ADO pipeline using Existing Azure Pipelines YAML file.
11. Manually run the pipeline and review each stage's logs.
12. Verify the actual image tag, Helm release, Kubernetes rollout,
    and application health using approved project-specific checks.
13. Locate real run status, run ID, stage results, and failure logs.
14. Explain common authentication, image pull, Helm, and timeout errors.

Use headings and practical examples where useful.
Do not claim that resources were created or a pipeline was run.
Do not invent successful results or application health endpoints.
Label any example result as hypothetical.
"""
        ),
        (
            "human",
            """
Project details:
{details}

Generated YAML:
{pipeline_yaml}

Provide the setup, execution, and result-verification guide.
"""
        )
    ])

    # 8. Second chain: generate the implementation guide
    setup_chain = setup_prompt | model | output_parser

    print("\nGenerating the ADO setup and verification guide...")

    setup_guide = setup_chain.invoke({
        "details": json.dumps(project_details, indent=2),
        "pipeline_yaml": pipeline_yaml
    })

    print("\n========== ADO IMPLEMENTATION GUIDE ==========\n")
    print(setup_guide)


if __name__ == "__main__":
    main()