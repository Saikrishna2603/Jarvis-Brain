from uuid import uuid4

from jarvis_platform.schemas.brain_orchestration import (
    BrainExecutionGraph,
    BrainExecutionGraphEdge,
    BrainExecutionGraphNode,
    BrainExecutionNodeType,
    BrainOrchestratorProposal,
)


class ExecutionGraphBuilder:
    """Build a non-executing DAG from an accepted brain proposal."""

    def build(self, proposal: BrainOrchestratorProposal) -> BrainExecutionGraph:
        graph_id = str(uuid4())
        nodes: list[BrainExecutionGraphNode] = [
            BrainExecutionGraphNode(
                node_id="intent",
                node_type=BrainExecutionNodeType.INTENT,
                label="Understand request",
            ),
            BrainExecutionGraphNode(
                node_id="context",
                node_type=BrainExecutionNodeType.CONTEXT,
                label="Assemble context",
            ),
        ]
        edges = [BrainExecutionGraphEdge(source="intent", target="context")]
        previous = "context"
        for step in proposal.plan_steps:
            node_id = step.step_id
            nodes.append(
                BrainExecutionGraphNode(
                    node_id=node_id,
                    node_type=step.node_type,
                    label=step.title,
                    metadata={
                        "action": step.action,
                        "target": step.target,
                        "requires_approval": step.requires_approval,
                    },
                )
            )
            dependencies = step.depends_on or [previous]
            for dependency in dependencies:
                edges.append(BrainExecutionGraphEdge(source=dependency, target=node_id))
            previous = node_id
        nodes.append(
            BrainExecutionGraphNode(
                node_id="verification",
                node_type=BrainExecutionNodeType.VERIFICATION,
                label="Verify result",
            )
        )
        edges.append(BrainExecutionGraphEdge(source=previous, target="verification"))
        nodes.append(
            BrainExecutionGraphNode(
                node_id="response",
                node_type=BrainExecutionNodeType.RESPONSE,
                label="Prepare response",
            )
        )
        edges.append(BrainExecutionGraphEdge(source="verification", target="response"))
        return BrainExecutionGraph(graph_id=graph_id, nodes=nodes, edges=edges)

