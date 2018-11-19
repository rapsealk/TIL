import java.rmi.Remote;
import java.rmi.RemoteException;

public interface RemoteMethodInvocation extends Remote {

    void printMessage() throws RemoteException;
}