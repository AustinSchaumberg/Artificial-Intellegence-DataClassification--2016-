import java.util.List;

public class DecTreeNodeImpl extends DecTreeNode
{
	DecTreeNodeImpl(String _label, String _attribute,
			String _parentAttributeValue, boolean _terminal)
	{
		super(_label, _attribute, _parentAttributeValue, _terminal);
		// TODO Auto-generated constructor stub
	}
	public String getNodeAttribute()
	{
		return this.attribute;
	}
	public void setParentValue(String _parentAttributeValue)
	{
		this.parentAttributeValue = _parentAttributeValue;
	}
	public boolean isTerminal()
	{
		return this.terminal;
	}
	public List<DecTreeNode> getListOfChildren()
	{
		return this.children;
	}
}
